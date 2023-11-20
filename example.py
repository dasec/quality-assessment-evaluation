"""
This is a CLI EDC plot example: python example.py
See the CLI help for more, as defined below.
"""

# Standard imports:
import argparse
from pathlib import Path
import json

# External imports:
import plotly as plt
import plotly.graph_objects as go
from tqdm import tqdm

# Local imports:
from edc import EdcErrorType
from edc import EdcSample
from edc import EdcSamplePair
from edc import EdcOutput
from edc import compute_edc
from edc import compute_edc_pauc
from edc import compute_edc_area_under_theoretical_best

comparison_type_to_error_type = {
    'mated': EdcErrorType.FNMR,
    'nonmated': EdcErrorType.FMR,
}


def main():
  # Parse CLI arguments:
  parser = argparse.ArgumentParser(
      prog='EDC example',
      description='This example computes EDC curves with pAUC values, and shows the plot in the default browser.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      '-d',
      '--data',
      type=Path,
      default=Path(__file__).parent / 'example_data.json',
      help='Similarity scores (either mated or non-mated) and quality scores as a JSON file.',
  )
  parser.add_argument(
      '-se',
      '--starting-error',
      type=float,
      default=0.05,
      help='The target starting error at the 0%% discard fraction.',
  )
  parser.add_argument(
      '-pauc',
      '--pauc-discard-limit',
      type=float,
      default=0.20,
      help='The upper discard limit used to compute the pAUC value of the EDC curves.',
  )
  parser.add_argument(
      '--shade-pauc',
      action=argparse.BooleanOptionalAction,
      default=True,
      help='Shade the pAUC for the best curve.',
  )
  parser.add_argument(
      '-norm',
      '--min-max-normalize',
      type=int,
      default=0,
      help='If a value above 0 is given, e.g. 100,'
      ' all quality scores will be normalized to the integer range [0, specified value]'
      ' by using min-max normalization.'
      ' Note that the minimum and maximum values are derived from the same data that is then normalized,'
      ' and this is only meant as an example for quality score normalization.',
  )
  args = parser.parse_args()

  # Load the input data:
  with open(args.data, 'r', encoding='utf-8') as file:
    data = json.load(file)

  comparison_type = data['similarity_scores']['type'].lower().replace('-', '')
  assert comparison_type in comparison_type_to_error_type, 'The comparisons must either be mated or non-mated.'

  similarity_scores = data['similarity_scores']['scores_by_pair_ids']
  quality_scores_per_algorithm = data['quality_scores_by_sample_id_per_algorithm']

  # Compute EDC curves:
  edc_outputs = {}
  for quality_assessment_algorithm, quality_scores in tqdm(quality_scores_per_algorithm.items(), desc='EDC curves'):
    # Apply the example min-max normalization if set (deactivated by default):
    if args.min_max_normalize > 0:
      quality_scores = _min_max_normalize(quality_scores, args.min_max_normalize)
    # Prepare sample and sample pair structures for compute_edc:
    samples = {sample_id: EdcSample(quality_score=quality_score) for sample_id, quality_score in quality_scores.items()}
    sample_pairs = []
    for pair_id, similarity_score in similarity_scores.items():
      sample_id1, sample_id2 = pair_id.split('-')
      sample_pairs.append(
          EdcSamplePair(
              samples=(
                  samples[sample_id1],
                  samples[sample_id2],
              ),
              similarity_score=similarity_score,
          ))

    # Run compute_edc:
    error_type = comparison_type_to_error_type[comparison_type]
    edc_output = compute_edc(
        error_type=error_type,
        sample_pairs=sample_pairs,
        starting_error=args.starting_error,
    )
    edc_outputs[quality_assessment_algorithm] = edc_output

  # Print true starting error vs. target starting error:
  print(f'- Target starting error (--starting-error CLI parameter): {args.starting_error}')
  some_edc_output = next(iter(edc_outputs.values()))
  true_starting_error = some_edc_output['error_fractions'][0]  # [0] for the 0% discard fraction.
  print(f'- True starting error at 0% discard fraction: {true_starting_error}')

  # The true starting error is equivalent for all QA algorithms, since it is at the 0% discard fraction:
  for edc_output in edc_outputs.values():
    assert true_starting_error == edc_output['error_fractions'][0]  # [0] for the 0% discard fraction.

  # Compute pAUC values for the EDC curves:
  pauc_values = {}
  for quality_assessment_algorithm, edc_output in edc_outputs.items():
    pauc_value = compute_edc_pauc(edc_output, args.pauc_discard_limit)
    pauc_value -= compute_edc_area_under_theoretical_best(edc_output, args.pauc_discard_limit)
    pauc_values[quality_assessment_algorithm] = pauc_value

  # Create the EDC plot:
  figure = _create_edc_plot(
      error_type=error_type,
      edc_outputs=edc_outputs,
      starting_error=true_starting_error,
      pauc_values=pauc_values,
      pauc_discard_limit=args.pauc_discard_limit,
      shade_pauc=args.shade_pauc,
  )
  figure.show()


def _min_max_normalize(quality_scores: dict, bin_count: int) -> dict:
  value_min = min(quality_scores.values())
  value_max = max(quality_scores.values())
  value_range = value_max - value_min
  return {
      sample_id: round(bin_count * ((quality_score - value_min) / value_range))
      for sample_id, quality_score in quality_scores.items()
  }


def _create_edc_plot(
    error_type: EdcErrorType,
    edc_outputs: dict,
    starting_error: float,
    pauc_values: dict,
    pauc_discard_limit: float,
    shade_pauc: bool = True,
) -> go.Figure():
  """Create a go.Figure() and plot the EDC curves, including the pAUC for the best curve."""
  figure = go.Figure()

  # Plot the constant starting error as a horizontal line:
  figure.add_trace(
      go.Scatter(
          x=[0, 1],
          y=[starting_error, starting_error],
          opacity=0.7,
          showlegend=False,
          line=dict(dash='dash', color='gray'),
      ))

  # Plot the 'theoretical best' line:
  figure.add_trace(
      go.Scatter(
          x=[0, starting_error],
          y=[starting_error, 0],
          opacity=0.7,
          showlegend=False,
          line=dict(dash='dash', color='gray'),
      ))

  # Plot the shaded pAUC for the best curve:
  if shade_pauc:
    best_edc_output = _get_best_edc_output(edc_outputs, pauc_values)
    _plot_shaded_pauc(
        figure=figure,
        edc_output=best_edc_output,
        pauc_discard_limit=pauc_discard_limit,
        starting_error=starting_error,
    )

  # Plot EDC curves, with labels showing the algorithm names, pAUC values, and relative rankings:
  relative_rankings = _compute_relative_rankings(pauc_values)
  for i, (quality_assessment_algorithm, edc_output) in enumerate(reversed(edc_outputs.items())):
    discard_fractions = edc_output['discard_fractions']
    error_fractions = edc_output['error_fractions']
    label = (f'{quality_assessment_algorithm}'
             f' | pAUC: {pauc_values[quality_assessment_algorithm]:.4f}'
             f' | Ranking: {relative_rankings[quality_assessment_algorithm]:.2f}')
    line_color = plt.colors.DEFAULT_PLOTLY_COLORS[i % len(plt.colors.DEFAULT_PLOTLY_COLORS)]
    figure.add_trace(
        go.Scatter(
            x=discard_fractions,
            y=error_fractions,
            name=label,
            line_shape='hv',  # 'hv' for stepwise interpolation.
            line_color=line_color,
        ))

  # Adjust the figure layout and return the figure:
  figure.update_layout(
      template='plotly_white',
      xaxis_title='Fraction of discarded comparisons',
      yaxis_title=error_type.value,
  )
  return figure


def _compute_relative_rankings(pauc_values: dict) -> dict:
  """Min-max normalize the pAUC values as 'relative rankings' (0 being the best algorithm, 1 being the worst)."""
  relative_rankings = {}
  pauc_value_min = min(pauc_values.values())
  pauc_value_max = max(pauc_values.values())
  pauc_value_range = pauc_value_max - pauc_value_min
  for quality_assessment_algorithm, pauc_value in pauc_values.items():
    relative_rankings[quality_assessment_algorithm] = (pauc_value - pauc_value_min) / pauc_value_range
  return relative_rankings


def _get_best_edc_output(edc_outputs: dict, pauc_values: dict) -> EdcOutput:
  """Get the best EdcOutput according to the pAUC values."""
  best_edc_output = None
  best_pauc_value = None
  for quality_assessment_algorithm, pauc_value in pauc_values.items():
    if (best_pauc_value is None) or (pauc_value < best_pauc_value):
      best_pauc_value = pauc_value
      best_edc_output = edc_outputs[quality_assessment_algorithm]
  return best_edc_output


def _plot_shaded_pauc(
    figure: go.Figure,
    edc_output: EdcOutput,
    pauc_discard_limit: float,
    starting_error: float,
):
  """Plot the pAUC in the given figure."""
  pauc_curve = {
      'x': edc_output['discard_fractions'],
      'y': edc_output['error_fractions'],
  }
  pauc_curve = _cut_curve(pauc_curve, x_limit=pauc_discard_limit)
  if pauc_discard_limit <= starting_error:
    curve_x_min = [0, pauc_discard_limit]
    curve_y_min = [starting_error, starting_error - pauc_discard_limit]
  else:
    curve_x_min = [0, starting_error, pauc_discard_limit]
    curve_y_min = [starting_error, 0, 0]
  for trace in _create_area_traces(pauc_curve['x'], pauc_curve['y'], curve_x_min, curve_y_min):
    figure.add_trace(trace)


def _cut_curve(curve: dict, x_limit: float):
  """Utility function to cut a curve for plotting purposes."""
  new_curve = {
      'x': [],
      'y': [],
  }
  index = -1
  while True:
    index += 1
    if index >= len(curve['x']):
      break
    x_value = curve['x'][index]
    if x_value <= x_limit:
      new_curve['x'].append(x_value)
      new_curve['y'].append(curve['y'][index])
      if x_value == x_limit:
        break
    else:
      if len(new_curve['x']) > 0:
        new_curve['x'].append(x_limit)
        new_curve['y'].append(new_curve['y'][-1])
      break
  return new_curve


def _create_area_traces(
    curve_x_max,
    curve_y_max,
    curve_x_min=None,
    curve_y_min=None,
    area_line_color='lightgray',
) -> list:
  """Utility function to create a shaded plot area."""
  traces = []
  if curve_y_min is not None:
    fill = 'tonexty'
  else:
    fill = 'tozeroy'
  if curve_y_min is not None:
    traces.append(go.Scatter(
        x=curve_x_min,
        y=curve_y_min,
        showlegend=False,
        line_color='rgba(0,0,0,0)',
    ))
  traces.append(
      go.Scatter(
          x=curve_x_max,
          y=curve_y_max,
          showlegend=False,
          line_color=area_line_color,
          line_shape='hv',
          fill=fill,
          fillpattern=go.scatter.Fillpattern(shape='.', solidity=0.01),
      ))
  return traces


if __name__ == '__main__':
  main()
