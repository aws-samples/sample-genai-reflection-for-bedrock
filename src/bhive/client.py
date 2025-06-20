"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

import functools
from typing import Callable

from botocore.config import Config

from bhive import chat, config, cost, inference, logger, struct_output
from bhive.evaluators import BudgetConfig, GridResults, TrialResult, answer_in_text
from bhive.utils import create_bedrock_client, parse_bedrock_output


class Hive:
    """
    A class for enhancing reasoning capabilities by leveraging multiple models
    and maintaining conversation history.

    Attributes:
        runtime_client (boto3.Client): The Boto3 client used to communicate
            with the Bedrock runtime service.

    Parameters:
        client_config (botocore.config.Config | None):
            Configuration for the Boto3 client. If provided, a new client
            will be created using this configuration.
        client (boto3.Client | None):
            An existing Boto3 client. If provided, it will be used directly
            instead of creating a new client. Only one of `client_config`
            or `client` should be provided.

    Raises:
        ValueError: If both `client_config` and `client` are provided, or if
            neither is provided.
    """

    def __init__(self, client_config: Config | None = None, client=None) -> None:
        """Initializes a Hive instance connected to a Boto3 client.

        This constructor either creates a new Boto3 client using the provided
        `client_config` or uses the existing `client` or trying from local environment.

        Parameters:
            client_config (botocore.config.Config | None): Configuration for the Boto3 client.
            client (boto3.Client | None): Existing Boto3 client.

        Raises:
            ValueError: If both `client_config` and `client` are provided
        """
        if client and client_config:
            raise ValueError("Only one of client or client_config should be provided.")
        if client_config:
            self.runtime_client = create_bedrock_client(client_config)
        elif client:
            if not hasattr(client, "converse"):
                raise ValueError("Provided client does not have a 'converse' method.")
            self.runtime_client = client
        else:
            self.runtime_client = create_bedrock_client()

    def converse(
        self, messages: list[dict], config: config.HiveConfig, **converse_kwargs
    ) -> chat.HiveOutput:
        """Invokes conversation with Hive using inference parameters.

        This method sends a message to the configured models and processes
        their responses based on the provided configuration.

        Parameters:
            messages (list[dict]): A list of Converse API formatted messages.
            config (config.HiveConfig): An inference configuration that outlines
                the models to be used, the number of rounds of reflection/debate,
                and the choice of aggregation model.
            converse_kwargs (dict): Additional keyword arguments to be passed to the converse method.

        Returns:
            chat.HiveOutput: A response object containing the answer (or answers if not aggregated) and the full chat history.
        """
        _all_models = config.bedrock_model_ids
        if config.aggregator_model_id:
            _all_models += [config.aggregator_model_id]

        # NOTE assumes first message holds the user prompt
        message = messages[0].get("content", [{}])[0].get("text", "")
        if config.output_model and message:
            struct_prompt = struct_output.prompt(config.output_model)
            messages[0]["content"][0]["text"] = message + f"\n{struct_prompt}"

        chatlog = chat.ChatLog(_all_models, messages, config.use_prompt_caching)
        logger.info(f"Starting inference with {config=} and {converse_kwargs=}")

        _converse_func = functools.partial(self._converse, **converse_kwargs)
        response: str | list[str] | None = None
        if config.single_model_single_call:
            # single model call
            response, chatlog = inference.single_model_single_call(config, chatlog, _converse_func)

        elif config.multi_model_single_call:
            # multi model / single round debate
            response, chatlog = inference.multi_model_single_call(
                config, chatlog, _converse_func, message
            )

        elif config.single_model_multi_call:
            # single model but reflection
            response, chatlog = inference.single_model_multi_call(
                config, chatlog, _converse_func, message
            )

        else:
            # multi model + multi round debate
            response, chatlog = inference.multi_model_multi_call(
                config, chatlog, _converse_func, message
            )

        # parsing structured outputs
        parsed_response = None
        if config.output_model:
            if isinstance(response, list):
                parsed_response = [struct_output.parse(r, config.output_model) for r in response]
            else:
                parsed_response = struct_output.parse(response, config.output_model)

        logger.info(f"Generated final {response=} and {parsed_response=}")
        return chat.HiveOutput(
            response=response,
            parsed_response=parsed_response,
            chat_history=chatlog.history,
            usage=chatlog.usage,
            metrics=chatlog.metrics,
            stopReason=chatlog.stopReason,
            trace=chatlog.trace,
            cost=cost.TotalCost(value=cost.calculate_cost(chatlog.usage)),
        )

    def _converse(
        self, model_id: str, messages: list[dict], **runtime_kwargs
    ) -> chat.ConverseResponse:
        try:
            response = self.runtime_client.converse(
                messages=messages,
                modelId=model_id,
                **runtime_kwargs,
            )
        except Exception as e:
            logger.error(
                f"Converse call failed for {model_id=} with {messages=} and {runtime_kwargs=}"
            )
            raise e
        status_code = response["ResponseMetadata"]["HTTPStatusCode"]
        if status_code != 200:
            logger.error(f"Converse call failed for {model_id=} with {status_code=}")
            converse_response = chat.ConverseResponse(answer="Failed to provide a response.")
        answer = parse_bedrock_output(response)
        converse_response = chat.ConverseResponse(
            answer=answer,
            usage=response["usage"],
            metrics=response["metrics"],
            stopReason=response["stopReason"],
            trace=response.get("trace", {}),
        )
        logger.debug(f"Received answer from {model_id}:\n{answer}")
        return converse_response

    def optimise(
        self,
        dataset: list[tuple[str, str]],
        trial_config: config.TrialConfig,
        budget_config: BudgetConfig | None = None,
        evaluator: Callable[[str, str], bool] = answer_in_text,
        **converse_kwargs,
    ) -> GridResults:
        """
        Optimises hyperparameters to find the best HiveConfig for the given dataset.

        This method performs a grid search over possible hyperparameter configurations
        defined in the provided `trial_config` and evaluates each configuration's
        performance using the specified `evaluator`. It tracks the best configuration
        based on a score and resource usage, and considers a budget constraint if provided.

        Parameters:
            dataset (list[tuple[str, str]]): A list of (input, expected_output) pairs
                                            used for evaluating the model.
            trial_config (config.TrialConfig): Configuration containing the possible
                                            hyperparameter settings to try.
            budget_config (BudgetConfig | None, optional): Optional configuration that
                                                        constrains the inference budget.
            evaluator (Callable[[str, str], bool], optional): A function that takes
                                                            a model's output and the expected
                                                            output to assess performance.
            **converse_kwargs: Additional keyword arguments passed to the conversation
                            evaluation process (e.g., specific model parameters).

        Returns:
            GridResults: An object containing the results of the grid search, including the best performing configuration and its evaluation score.
        """
        logger.info("Starting optimisation ...")
        configs = trial_config._all_configuration_options()
        logger.info(f"Optimising over {len(configs)} configurations")
        results = GridResults()
        if budget_config:
            cost_dict = budget_config.cost_dictionary
        else:
            logger.warning("Using built-in cost dictionary, which may be out of date")
            cost_dict = cost.MODELID_COSTS_PER_TOKEN

        for _config in configs:
            # TODO implement smarter stateful pruning of configs
            ## e.g. if a model is too expensive at 1 round of reflection, exclude anymore
            try:
                candidate = self._objective(
                    dataset, _config, evaluator, cost_dict, **converse_kwargs
                )
                results.individual_results.append(candidate)

                if (
                    not results.best
                    or results.best_score(candidate)
                    or results.better_resource_usage(candidate)
                ):
                    # good candidate
                    if budget_config and not budget_config.check_budget(candidate):
                        continue
                    results.best = candidate

            except Exception as e:
                logger.error(f"Error during optimisation: {e}")
                continue
        return results

    def _objective(
        self,
        dataset: list[tuple[str, str]],
        hive_config: config.HiveConfig,
        evaluator: Callable[[str, str], bool],
        cost_dictionary: dict[str, cost.TokenPrices],
        **converse_kwargs,
    ) -> TrialResult:
        """Objective function to optimize the inference method."""
        cumulative_eval_score = 0
        costs = []
        latencies = []
        for message, expected_response in dataset:
            try:
                messages = [{"role": "user", "content": [{"text": message}]}]
                output = self.converse(messages, hive_config, **converse_kwargs)
                answer = output.response
                costs.append(cost.calculate_cost(output.usage, cost_dictionary, strict=True))
                latencies.append(cost.average_latency(output.metrics))
                if not isinstance(answer, str):
                    raise TypeError("Optimisation must be performed with single responses.")
                eval_score = evaluator(expected_response, answer)
                if eval_score:
                    cumulative_eval_score += eval_score
            except Exception as e:
                logger.error(f"Error during sample inference: {e}")
                continue
        return TrialResult(
            score=cumulative_eval_score / len(dataset),
            config=hive_config,
            avg_latency_seconds=sum(latencies) / len(latencies),
            avg_cost_dollars=sum(costs) / len(costs),
        )
