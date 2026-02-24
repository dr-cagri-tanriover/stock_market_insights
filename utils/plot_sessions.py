from pathlib import Path
import matplotlib.pyplot as plt

"""
attributes_default_template=
{
    'save_folder': None, # or Path object
    'display_plots': False,
    'gridOn': True,
    'legendOn': True,
    'x_label': "x axis label",
    'y_label': "y axis label",
    'title': "plot title",
    'color': 'b',
    'marker': 'o',
    'marker_size': 2,
    'figsize'=(12, 6),
    'filename': None or string
}
"""

class BasePlotSessions:
    """
    Base class for plot sessions.
    """
    def __init__(self, attributes: dict):
        self.attr = {}
        self.attr['save_folder'] = attributes['save_folder']  # Folder to save the plots (can be None to skip saving)
        self.attr['display_plots'] = attributes['display_plots']  # Bool: True to display plots, False not to display
        self.attr['gridOn'] = attributes['gridOn']  # Bool: True to show grid, False not to show
        self.attr['legendOn'] = attributes['legendOn']  # Bool: True to show legend, False not to show
        self.attr['figsize'] = attributes['figsize']  # Figure size for the plot

    def get_default_attribs_dict(self):

        return {
        'save_folder': None,
        'display_plots': False,
        'gridOn': True,
        'legendOn': True,
        'figsize': (12, 6),
        'x_label': "x axis label",
        'y_label': "y axis label",
        'title': "plot title",
        'color': 'b',
        'marker': 'o',
        'marker_size': 2,
        'filename': None
        }

    def clean_up(self):
        # Close all matplotlib figures to free up memory
        plt.close('all')
        print("All plot windows closed.")


    def enable_interactive_plots(self):
        plt.ion()
        plt.show(block=False)
        # Give matplotlib time to render the plot window
        plt.pause(0.1)  # Brief pause to ensure plot window is rendered
        plt.draw()  # Force a draw to update the display


class scatterPlot2D(BasePlotSessions):
    """
    Handles a simple 2D scatter plot.
    """
    def __init__(self, attributes: dict):
        default_attribs = self.get_default_attribs_dict()
        attributes = {**default_attribs, **attributes}  # New values override default values, if they exist. Otherwise, default values persist.

        super().__init__(attributes)
        self.attr['x_label'] = attributes['x_label']
        self.attr['y_label'] = attributes['y_label']
        self.attr['title'] = attributes['title']
        self.attr['color'] = attributes['color']
        self.attr['marker'] = attributes['marker']
        self.attr['marker_size'] = attributes['marker_size']
        self.attr['filename'] = attributes['filename']

    def update_current_attributes_dict(self, new_attributes: dict):
        self.attr = {**self.attr, **new_attributes}  # Only the keys in new_attributes override the keys in self.attr

    def plot(self, x_data=None, y_data=None, traces=None, label=None):
        """
        Plot one or more scatter traces on the same axes.

        Single trace (original usage):
            plot(x_data, y_data)
            plot(x_data, y_data, label='Series 1')

        Multiple traces:
            plot(traces=[
                {'x': x1, 'y': y1, 'label': 'A'},
                {'x': x2, 'y': y2, 'label': 'B', 'color': 'r', 'marker': 's'},
            ])
        Each trace dict can override 'color', 'marker', 'marker_size' per trace; defaults come from self.attr.
        """
        fig = plt.figure(figsize=self.attr['figsize'])
        if traces is not None:
            for tr in traces:
                x, y = tr['x'], tr['y']
                color = tr.get('color', self.attr['color'])
                marker = tr.get('marker', self.attr['marker'])
                size = tr.get('marker_size', self.attr['marker_size'])
                lbl = tr.get('label', None)
                plt.scatter(x, y, color=color, marker=marker, s=size, label=lbl)
        else:
            if x_data is None or y_data is None:
                raise ValueError("Either (x_data, y_data) or traces= must be provided.")
            plt.scatter(x_data, y_data, color=self.attr['color'], marker=self.attr['marker'],
                        s=self.attr['marker_size'], label=label)
        plt.xlabel(self.attr['x_label'])
        plt.ylabel(self.attr['y_label'])
        plt.title(self.attr['title'])
        plt.grid(self.attr['gridOn'])
        if self.attr['legendOn']:
            plt.legend()
        plt.tight_layout()

        if self.attr['display_plots']:
            self.enable_interactive_plots()

        if self.attr['save_folder'] is not None and self.attr['filename'] is not None:
            self.save_plot()

        self.clean_up()


    def save_plot(self):
        # 'save_folder' and 'filename' presence already checked by the caller.    
        self.attr['save_folder'].mkdir(parents=True, exist_ok=True)           # Create the folder if it doesn't exist
        plt.savefig(self.attr['save_folder'] / self.attr['filename'])  # save_folder is expected to be a Path object. filename is a string