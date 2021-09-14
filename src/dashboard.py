import dash
from dash.exceptions import DashException
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from dash_bootstrap_components._components.Button import Button
from src.experiments import polynomial, sampler_trace, synthetic
from src.experiments.common import ExperimentHandler

ALL_EXPERIMENTS = [polynomial, sampler_trace, synthetic]


def get_handlers():
    experiment_handlers = {h.name : h for h in (ExperimentHandler(e) for e in ALL_EXPERIMENTS)}
    return experiment_handlers


app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])

sidebar = html.Div(
    [
        html.P("Collection of various experiments", className="lead"),
        dbc.Nav(
            [dbc.NavLink("Home", href="/", active="exact")]
            + [dbc.NavLink(h, href=f"/{h}", active="exact") for h in get_handlers()],
            vertical=True,
            pills=True,
        ),
    ],
)

content = html.Div(id="page-content")

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dbc.Container(
            [
                dbc.Row(
                    [
                        html.H2("Experiments", className="display-4"),
                        html.Hr(),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(sidebar, width=4, sm=3),
                        dbc.Col(dbc.Container(content), width=8),
                    ]
                ),
            ],
            fluid=False,
        ),
    ]
)


def get_experiment_content(experiment):

    results = experiment.results()
    fig = experiment.plots()
    fig = theme_plot(fig)

    return html.Div(
        [
            dbc.Row([dbc.ButtonGroup([dbc.Button("Redo Experiment")])], justify="end"),
            dbc.Row([dcc.Graph(figure=fig, style={"width": "100%"})]),
        ]
    )


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_experiment(pathname):

    if pathname == "/":
        return html.P("This is the content of the home page!")
    # elif pathname[1:] in experiments:
    #     return get_experiment_content(experiments[pathname[1:]])
    else:
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )


if __name__ == "__main__":
    app.run_server(debug=True)
