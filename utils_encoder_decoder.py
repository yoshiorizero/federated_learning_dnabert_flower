def prepare_encoder_decoder_model_kwargs(**kwargs):
    """ Prepare the encoder and decoder's keyword arguments.

    Keyword arguments come in 3 flavors:
    - encoder-specific (prefixed by `encoder_`)
    - decoder-specific (prefixed by `decoder_`)
    - those that apply to the model as whole.

    We let the specific kwargs override the common ones in case of
    conflict.
    """

    kwargs_common = {
        argument: value
        for argument, value in kwargs.items()
        if not argument.startswith("encoder_") and not argument.startswith("decoder_")
    }
    if "input_ids" in kwargs_common:
        kwargs["encoder_input_ids"] = kwargs_common.pop("input_ids")

    decoder_kwargs = kwargs_common.copy()
    encoder_kwargs = kwargs_common.copy()
    encoder_kwargs.update(
        {argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")}
    )
    decoder_kwargs.update(
        {argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")}
    )
    decoder_kwargs["encoder_attention_mask"] = encoder_kwargs.get("attention_mask", None)
    return encoder_kwargs, decoder_kwargs
