import pytest
from pydantic import ValidationError
from bhive import BudgetConfig, TokenPrices

test_costs = {
    "model_a": TokenPrices(input_per_1000=0.1, output_per_1000=0.1),
    "model_b": TokenPrices(input_per_1000=0.2, output_per_1000=0.2),
}


# Use pytest fixture with mocker to mock the cost.MODELID_COSTS_PER_TOKEN
@pytest.fixture
def mock_cost(mocker):
    mocker.patch("bhive.cost.MODELID_COSTS_PER_TOKEN", test_costs.copy())
    return test_costs


def should_update_existing_keys_in_cost_dictionary(mock_cost):
    new_costs = {
        "model_a": TokenPrices(
            input_per_1000=0.15, output_per_1000=0.15, currency="USD"
        ),  # Update the price of model_a
    }
    budget_config = BudgetConfig(
        max_dollar_per_sample=10.0, max_seconds_per_sample=5.0, cost_dictionary=new_costs
    )
    assert budget_config.cost_dictionary["model_a"].input_per_1000 == 0.15
    assert (
        budget_config.cost_dictionary["model_b"].input_per_1000 == 0.2
    )  # model_b should remain unchanged


def should_add_new_keys_to_cost_dictionary(mock_cost):
    new_costs = {
        "model_c": TokenPrices(
            input_per_1000=0.3, output_per_1000=0.3, currency="USD"
        ),  # New model 'model_c'
    }
    budget_config = BudgetConfig(
        max_dollar_per_sample=10.0, max_seconds_per_sample=5.0, cost_dictionary=new_costs
    )
    assert "model_c" in budget_config.cost_dictionary
    assert budget_config.cost_dictionary["model_c"].input_per_1000 == 0.3


def should_raise_validation_error_for_invalid_cost_dictionary():
    invalid_costs = {"model_a": "Invalid negative cost"}
    with pytest.raises(ValidationError):
        _ = BudgetConfig(
            max_dollar_per_sample=10.0, max_seconds_per_sample=5.0, cost_dictionary=invalid_costs
        )


def should_raise_validation_error_for_invalid_token_prices():
    with pytest.raises(ValidationError):
        _ = TokenPrices(input_per_1000=-0.1, output_per_1000=0.1)
