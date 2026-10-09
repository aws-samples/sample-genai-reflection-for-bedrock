from bhive import xml_output


def should_extract_simple_tag():
    assert xml_output.extract("<answer>42</answer>") == "42"


def should_extract_last_occurrence_when_repeated():
    text = "Let me think <answer>wrong</answer> ... actually <answer>right</answer>"
    assert xml_output.extract(text) == "right"


def should_be_case_insensitive_and_strip():
    assert xml_output.extract("<ANSWER>  Paris \n</ANSWER>") == "Paris"


def should_support_custom_tag():
    assert xml_output.extract("<label>positive</label>", tag="label") == "positive"


def should_return_default_when_absent():
    assert xml_output.extract("no tags here", default="N/A") == "N/A"


def should_handle_truncated_open_tag():
    assert xml_output.extract("the result is <answer>partial") == "partial"


def should_extract_all_pairs_in_order():
    text = "<f>a</f> middle <f>b</f>"
    assert xml_output.extract_all(text, "f") == ["a", "b"]


def should_build_instruction_with_tag():
    msg = xml_output.instruction("sql")
    assert "<sql></sql>" in msg
