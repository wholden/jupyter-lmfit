import numpy as np
import ipywidgets as widgets
from IPython.display import clear_output, HTML
from IPython import get_ipython
import matplotlib.pyplot as plt
import lmfit


def _get_all_models():
    models = {}
    for sc in lmfit.model.Model.__subclasses__():
        if sc.__module__ == 'lmfit.models':
            models[sc.__name__] = sc
    models.pop('PolynomialModel')  # PolynomialModel not supported yet, requires a degree to be specified
    models.pop('SplitLorentzianModel')  # Throws error
    return models


def _update_parameter(old, new):
    """Couldn't find a way built into lmfit to update an existing
    parameter with the values from a new one. This is necessary to
    maintain the link between the display widget and the backend
    parameters objects."""
    old.set(
        value=new.value,
        vary=new.vary,
        min=new.min,
        max=new.max,
        expr=new.expr,
        brute_step=new.brute_step
    )


def _update_params(old, new):
    for n, p in new.items():
        _update_parameter(old[n], p)


class ParameterWidgetGroup(object):
    """Modified from existing lmfit.ui.ipy_fitter"""
    def __init__(self, par):
        self.par = par
        widgetlayout = {'flex': '1 1 auto', 'width': 'auto', 'margin': '0px 0px 0px 0px'}
        width = {'description_width': '10px'}

        # Define widgets.
        self.value_text = widgets.FloatText(description=par.name, style={'description_width': '130px'}, layout={'flex': '2 1 auto', 'width': 'auto', 'margin': 'auto 0px auto auto'})
        self.min_text = widgets.FloatText(style=width, layout=widgetlayout)
        self.max_text = widgets.FloatText(style=width, layout=widgetlayout)
        self.min_checkbox = widgets.Checkbox(description='min', style=width, layout=widgetlayout)
        self.max_checkbox = widgets.Checkbox(description='max', style=width, layout=widgetlayout)
        self.vary_checkbox = widgets.Checkbox(description='vary', style=width, layout=widgetlayout)

        # Set widget values and visibility.
        if par.value is not None:
            self.value_text.value = self.par.value
        min_unset = self.par.min is None or self.par.min == -np.inf
        max_unset = self.par.max is None or self.par.max == np.inf
        self.min_checkbox.value = not min_unset
        self.min_text.value = self.par.min
        self.min_text.disabled = min_unset
        self.max_checkbox.value = not max_unset
        self.max_text.value = self.par.max
        self.max_text.disabled = max_unset
        self.vary_checkbox.value = self.par.vary

        # Configure widgets to sync with par attributes.
        self.value_text.observe(self._on_value_change, names='value')
        self.min_text.observe(self._on_min_value_change, names='value')
        self.max_text.observe(self._on_max_value_change, names='value')
        self.min_checkbox.observe(self._on_min_checkbox_change, names='value')
        self.max_checkbox.observe(self._on_max_checkbox_change, names='value')
        self.vary_checkbox.observe(self._on_vary_change, names='value')

    def _on_value_change(self, change):
        self.par.set(value=change['new'])

    def _on_min_checkbox_change(self, change):
        self.min_text.disabled = not change['new']
        if not change['new']:
            self.min_text.value = -np.inf

    def _on_max_checkbox_change(self, change):
        self.max_text.disabled = not change['new']
        if not change['new']:
            self.max_text.value = np.inf

    def _on_min_value_change(self, change):
        if not self.min_checkbox.disabled:
            self.par.set(min=change['new'])

    def _on_max_value_change(self, change):
        if not self.max_checkbox.disabled:
            self.par.set(max=change['new'])

    def _on_vary_change(self, change):
        self.par.set(vary=change['new'])

    def close(self):
        # one convenience method to close (i.e., hide and disconnect) all
        # widgets in this group
        self.value_text.close()
        self.min_text.close()
        self.max_text.close()
        self.vary_checkbox.close()
        self.min_checkbox.close()
        self.max_checkbox.close()

    def get_widget(self):
        box = widgets.HBox([self.value_text, self.vary_checkbox,
                            self.min_checkbox, self.min_text,
                            self.max_checkbox, self.max_text])
        return box

    # Make it easy to set the widget attributes directly.
    @property
    def value(self):
        return self.value_text.value

    @value.setter
    def value(self, value):
        self.value_text.value = value

    @property
    def vary(self):
        return self.vary_checkbox.value

    @vary.setter
    def vary(self, value):
        self.vary_checkbox.value = value

    @property
    def min(self):
        return self.min_text.value

    @min.setter
    def min(self, value):
        self.min_text.value = value

    @property
    def max(self):
        return self.max_text.value

    @max.setter
    def max(self, value):
        self.max_text.value = value

    @property
    def name(self):
        return self.par.name


class LmfitWidget:

    def __init__(self, data, x):
        self.data = data
        self.x = x
        self.mod = None
        self.pars = None
        self.paramwidgetscontainer = None
        self.fit = None

    def add_model(self, model):
        if self.mod is None:
            self.mod = model(prefix='{}{}_'.format(model().name[6:-1], 0))
        else:
            i = 1
            added = False
            while not added:
                try:
                    self.mod += model(prefix='{}{}_'.format(model().name[6:-1], i))
                    added = True
                except NameError:
                    i += 1
        newpars = self.mod.make_params()
        if self.pars is None:
            self.pars = newpars
        else:
            oldpars = self.pars.copy()
            self.pars = newpars
            _update_params(self.pars, oldpars)

    def clear_models(self):
        self.mod = None
        self.pars = None
        self.fit = None

    def make_params_widget(self):
        box_layout = widgets.Layout(display='flex',
                                    flex_flow='column',
                                    align_items='stretch',
                                    width='100%')
        if self.paramwidgetscontainer is not None:
            if self.pars is not None:
                self.paramwidgets = [ParameterWidgetGroup(p) for _, p in self.pars.items()]
                for pw in self.paramwidgets:
                    pw.value_text.observe(lambda e: self.update_plot(), names='value')
                sortkeys = np.argsort(list(self.pars.keys()))  # ugly solution
                self.paramwidgetscontainer.children = [self.paramwidgets[i].get_widget() for i in sortkeys]  # [pw.get_widget() for pw in self.paramwidgets]
            else:
                self.paramwidgetscontainer.children = []
        else:
            self.paramwidgetscontainer = widgets.VBox([], layout=box_layout)
            return self.paramwidgetscontainer

    def make_models_widget(self):
        self.modelsdropdown = widgets.Dropdown(
            options=_get_all_models(),
            description='Model:',
            disabled=False,
        )
        self.addmodelbutton = widgets.Button(
            description='Add Model',
            disabled=False,
            button_style='',
            tooltip='Click to add another model',
            icon='plus'
        )
        self.clearmodelsbutton = widgets.Button(
            description='Clear Models',
            disabled=False,
            button_style='info',
            tooltip='Click to clear all models',
            icon='minus'
        )
        self.addmodelbutton.on_click(self._handle_addmodel_button)
        self.clearmodelsbutton.on_click(self._handle_clearmodel_button)

        return widgets.HBox(
            children=[self.modelsdropdown, self.addmodelbutton, self.clearmodelsbutton],
            layout={'height': '50px'}
        )

    def fitting_widget(self):
        self.runfitbutton = widgets.Button(
            description='Run Fit',
            disabled=False,
            button_style='success',
            tooltip='Click to run fit',
            icon=''
        )
        # self.guessparamsbutton = widgets.Button(
        #     description='Guess Parameters',
        #     disabled=False,
        #     button_style='info',
        #     tooltip='Click to guess parameters',
        #     icon=''
        # )
        # self.guessparamsbutton.on_click(self._handle_guessparams_button)
        self.runfitbutton.on_click(self._handle_runfit_button)
        return widgets.HBox(
            children=[self.runfitbutton],  # self.guessparamsbutton],
            layout={'height': '50px'}
        )

    def _handle_addmodel_button(self, event):
        self.fit = None
        self.add_model(self.modelsdropdown.value)
        self.make_params_widget()
        self.update_plot()

    def _handle_clearmodel_button(self, event):
        self.clear_models()
        self.make_params_widget()
        self.update_plot()

    def _handle_guessparams_button(self, event):
        updatedpars = self.mod.guess(self.data, x=self.x)
        _update_params(self.pars, updatedpars)
        self.make_params_widget()
        self.update_plot()

    def _handle_runfit_button(self, event):
        self.runfitbutton.description = 'Running...'
        self.runfitbutton.button_style = 'warning'
        self.fit = self.mod.fit(self.data, self.pars, x=self.x)  # , nan_policy='omit')
        _update_params(self.pars, self.fit.params)
        self.make_params_widget()
        self.update_plot()
        self.runfitbutton.description = 'Run Fit'
        self.runfitbutton.button_style = 'success'

    def _sync_pars(self):
        """Propagate backend changes to parameters to widget display."""
        raise NotImplementedError

    def plot_widget(self):
        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.x, self.data, label='data')
        self.ax.plot(self.x, np.zeros_like(self.x), label='model')
        self.ax.legend()
        return self.fig.canvas

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.x, self.data, label='data')

        numsteps = 1000
        xmod = np.linspace(self.x.min(), self.x.max(), numsteps)
        if self.mod is None:
            self.ax.plot(xmod, np.zeros_like(xmod), label='model')
        else:
            try:
                self.ax.plot(xmod, self.fit.eval(params=self.pars, x=xmod), label='model')
                for c, v in self.fit.eval_components(params=self.pars, x=xmod).items():
                    plt.plot(xmod, v, '--', label=c)
            except AttributeError:
                self.ax.plot(xmod, self.mod.eval(params=self.pars, x=xmod), label='model')
                for c, v in self.mod.eval_components(params=self.pars, x=xmod).items():
                    plt.plot(xmod, v, '--', label=c)

        self.ax.legend()
        self.fig.canvas.draw()
#         self.fig.canvas.flush_events()

    def render(self):
        return widgets.VBox(
            children=[
                self.make_models_widget(), 
                self.make_params_widget(), 
                self.fitting_widget(),
                self.plot_widget()
            ])

    def show_fit_params(self):
        if self.fit is None:
            raise RuntimeError('Must have a completed fit in order to show the parameters.')
        return HTML(html_repr(self.fit.params))


# Temporary inclusion until this merges into lmfit/main
def html_repr(self):
    """Returns a HTML representation of parameters data."""

    # We omit certain columns if they would be empty.
    any_err = any([p.stderr is not None for p in self.values()])
    any_par_expr = any([p.expr is not None for p in self.values()])
    # Helper functions for shorter code
    html = []
    add = html.append
    cell = lambda x: add('<td>%s</td>' % x)
    head = lambda x: add('<th>%s</th>' % x)

    add('<table>')
    # Header ----
    add('<tr>')
    headers = ['name', 'value', 'min', 'max', 'vary']  # ['name', 'value', '(init)', 'min', 'max', 'vary']
    if any_err:
        headers = headers[:2] + ['', 'error', 'rel. error'] + headers[2:]
    if any_par_expr:
        headers.append('expr.')
    for h in headers:
        head(h)
    add('</tr>')
    # Parameters ----------
    for p in self.values():
        add('<tr>')
        cell(p.name)
        cell('%f' % p.value)
        if p.stderr is not None:
            cell('+/-')
            cell('%f' % p.stderr)
            cell('%.2f%%' % (100 * p.stderr / p.value))
        elif any_err:
            cell('')
            cell('')
            cell('')
        # cell(p.user_value)
        cell('%f' % p.min)
        cell('%f' % p.max)
        cell('%s' % p.vary)
        if p.expr is not None:
            cell('%s' % p.expr)
        elif any_par_expr:
            cell('')
        add('</tr>')
    add('</table>')
    return ''.join(html)
