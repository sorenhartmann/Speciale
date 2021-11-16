





_DEFAULT_PRIORS = PriorSpec(NormalPrior())


class ModuleWithPrior(nn.Module):

    priors: nn.ModuleDict

    def __init__(self, module, priors: nn.ModuleDict):

        super().__init__()
        self.module = module
        self.priors = priors

    def forward(self, x):
        return self.module.forward(x)

    def prior_log_prob(self: nn.Module):
        return sum(
            prior.log_prob(getattr(self.module, name)).sum()
            for name, prior in self.priors.items()
        )


def with_priors(module: nn.Module, prior_specs):

    module_class = type(module)

    if module_class not in prior_specs:
        return

    priors = nn.ModuleDict()
    for parameter_name, _ in module.named_parameters():
        priors[parameter_name] = prior_specs[module_class, parameter_name]

    return ModuleWithPrior(module, priors)


class ProbabilisticModel(Model):

    model: Model
    submodules_with_prior: List[ModuleWithPrior]

    def __init__(self, model, submodules_with_prior):

        super().__init__()
        self.model = model
        self.submodules_with_prior = submodules_with_prior

    def observation_model_gvn_output(self, output):
        return self.model.observation_model_gvn_output(output)

    def observation_model(self, input):
        return self.model.observation_model(input)

    def loss(self, output, target):
        return self.model.loss(output, target)

    def get_metrics(self):
        return self.model.get_metrics()

    def predict(self, x):
        return self.model.predict(x)

    def forward(self, x):
        return self.model.forward(x)

    def log_prior(self):
        """Returns p(theta)"""
        return sum(m.prior_log_prob() for m in self.submodules_with_prior)

    def log_likelihood(self, x, y):
        """Returns log p(y | x, theta)"""
        return self.observation_model(x).log_prob(y)





def as_probabilistic_model(model: Model, prior_spec=None):

    if prior_spec is None:
        prior_spec = _DEFAULT_PRIORS



    submodules_with_prior = []

    def replace_submodules_(module: nn.Module):

        helper = ModuleAttributeHelper(module)
        for key, child in helper.keyed_children():
            if type(child) in prior_spec:
                new_module = with_priors(child, prior_spec)
                helper[key] = new_module
                submodules_with_prior.append(new_module)
            else:
                replace_submodules_(child)

    replace_submodules_(model)

    return ProbabilisticModel(model, submodules_with_prior)
