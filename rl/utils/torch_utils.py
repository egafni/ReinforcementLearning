import pandas
import torch

# pandas.set_option("display.width", 1000)


def summarize(module, input_shape):
    """
    Summarizes a pytorch module

    :param module: a pytorch module
    :param input_shape: the input shape of the module
    :return: a pandas DataFrame of all the layers
    """

    def make_hook(module, input, output):
        """
        creates a hook on module which adds debug information to the data frame df
        """

        def get_shape(x):
            if isinstance(x, (list, tuple)):
                if len(x) > 1:
                    return ", ".join(get_shape(e) for e in x)
                else:
                    x = x[0]
            x = list(x.shape)
            x[0] = -1
            return str(tuple(x))

        df.loc[id(module), "input"] = get_shape(input)
        df.loc[id(module), "output"] = get_shape(output)

        params = 0
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += torch.prod(torch.LongTensor(list(module.weight.size())))
            df.loc[id(module), "trainable"] = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += torch.prod(torch.LongTensor(list(module.bias.size())))
        df.loc[id(module), "n_params"] = int(params)

    def add_hooks(module, name=None, depth=0):
        """
        add hooks for all layers
        """
        n_children = len(list(module.named_children()))
        yield dict(
            id=id(module),
            depth=depth,
            name=name,
            cls=module.__class__.__name__,
            call="" if len(list(module.named_children())) else str(module),
            hook=module.register_forward_hook(make_hook),
            n_children=n_children,
        )

        for name, m in module.named_children():
            for rv in add_hooks(m, name, depth=depth + 1):
                rv["parent_id"] = id(module)
                yield rv

    df = pandas.DataFrame(add_hooks(module)).set_index("id")
    # initialize these columns so make_hook can add things appropriately
    df["input"] = None
    df["output"] = None
    df["trainable"] = None
    df["n_params"] = None

    # run all the hooks
    try:
        input_shape = [2] + list(input_shape)

        module(torch.ones(input_shape))
    except:  # noqa
        # print the layers so far if we hit an exception for debugging purposes
        df = df.reset_index()
        for col in ["id", "parent_id", "hook"]:
            del df[col]
        print(df)
        raise

    # remove hooks for module
    for h in df["hook"]:
        h.remove()
    del df["hook"]  # delete column

    return df


def net_to_device(net):
    return "cuda" if next(net.parameters()).is_cuda else "cpu"
