def setup_altair():

    import os
    import altair as alt
    from toolz.curried import pipe

    def json_dir(data, data_dir='altairdata'):
        os.makedirs(data_dir, exist_ok=True)
        return pipe(data, alt.to_json(filename=data_dir + '/{prefix}-{hash}.{extension}') )

    alt.data_transformers.register('json_dir', json_dir)
    alt.data_transformers.enable('json_dir')