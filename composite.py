from NN import Model

class CompositeModel(Model):
    @staticmethod
    def get_options():
        ops = Model.get_options()
        ops.add(
            name="models",
            required=True,
            value_type=list,
            description="""The models to be composed."""
        )
        return ops

    def init_params(self):
        models = self.options.get('models')

        #The model_supports_batch has to be inferred after the models are set
        supports_batch = True
        for model in models:
            try:
                model_supports_batch = model.options.get('model_supports_batch')
                if not model_supports_batch:
                    supports_batch = False
                    break
            except KeyError:
                supports_batch = False
                break
        self.options.add(
            name="model_supports_batch",
            value=supports_batch,
            readonly=True,
            description="""Tells if this model supports batch feedforward"""
        )

        params = []

        for model in models:
            params += model.params

        return params

    def generate_input(self):
        return self.options.get('models')[0].input

    def generate_output(self, inputs):
        models = self.options.get('models')

        result = models[0].output

        for i in range(1,len(models)):
            result = models[i].generate_output([result])
        return result
