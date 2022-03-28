from collections import OrderedDict

from torch import nn

from oracle.utils.torch_utils import summarize


def test_summarize():
    model = nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 3, 3)),
                ("relu1", nn.ReLU()),
                ("conv2", nn.Conv2d(3, 2, 2)),
                ("relu2", nn.ReLU()),
                ("seq", nn.Sequential(nn.Flatten(), nn.Linear(14, 1))),
            ]
        )
    )

    df = summarize(model, (1, 10, 4))
    df = df.reset_index()
    del df["id"]
    del df["parent_id"]

    # to generate/replace this string run pytest with --pdb and print(df.to_string())
    #     assert (
    #         df.to_string()
    #         == """    name         cls  depth  n_children           input         output trainable n_params
    # 0   None  Sequential      0           5  (-1, 1, 10, 4)        (-1, 1)      None        0
    # 1  conv1      Conv2d      1           0  (-1, 1, 10, 4)  (-1, 3, 8, 2)      True       30
    # 2  relu1        ReLU      1           0   (-1, 3, 8, 2)  (-1, 3, 8, 2)      None        0
    # 3  conv2      Conv2d      1           0   (-1, 3, 8, 2)  (-1, 2, 7, 1)      True       26
    # 4  relu2        ReLU      1           0   (-1, 2, 7, 1)  (-1, 2, 7, 1)      None        0
    # 5    seq  Sequential      1           2   (-1, 2, 7, 1)        (-1, 1)      None        0
    # 6      0     Flatten      2           0   (-1, 2, 7, 1)       (-1, 14)      None        0
    # 7      1      Linear      2           0        (-1, 14)        (-1, 1)      True       15"""
    #     )

    return df
