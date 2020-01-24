#!/usr/local/bin/bash
__author__ = "xiangwang"
import os
import re


def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines


def ensure_dir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def uni2str(unicode_str):
    return str(unicode_str.encode("ascii", "ignore")).replace("\n", "").strip()


def has_numbers(input_string):
    return bool(re.search(r"\d", input_string))


def del_multichar(input_string, chars):
    for ch in chars:
        input_string = input_string.replace(ch, "")
    return input_string


def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def early_stopping(
    log_value, best_value, stopping_step, expected_order="acc", flag_step=100
):
    # early stopping strategy:
    assert expected_order in ["acc", "dec"]

    if (expected_order == "acc" and log_value >= best_value) or (
        expected_order == "dec" and log_value <= best_value
    ):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print(
            "Early stopping is trigger at step: {} log:{}".format(flag_step, log_value)
        )
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
    return model


def print_dict(dic):
    """print dictionary using specified format

    example: {"a": 1, "b": 2}
    output:
            "a": 1
            "b": 2
    """
    print("\n".join("{:10s}: {}".format(key, values) for key, values in dic.items()))
