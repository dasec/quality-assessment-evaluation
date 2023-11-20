"""
"Error versus Discard Characteristic" (EDC) functionality.

See example.py for a concrete usage example.
"""

# Standard imports:
from typing import Optional, Union, TypedDict, Callable, Iterable
from enum import Enum

# External imports:
import numpy as np

QualityScore = float
"""A quality score.
Higher values should canonically indicate higher utility (i.e. higher should mean better).
"""

SimilarityScore = float
"""A biometric comparison score that more specifically is a similarity score,
meaning that higher values indicate higher similarity between the compared biometric samples.
"""

SimilarityScores = Union[SimilarityScore, Iterable[SimilarityScore]]
"""One or multiple similarity scores."""


class EdcSample(TypedDict):
  """The input data for one sample that is required to compute an EDC curve,
  which is just the sample's quality score.
  """
  quality_score: QualityScore
  """The sample's quality score.
  In context of the EDC curve computation, a higher quality score is assumed to indicate higher utility.
  For a typical EDC setup, samples with lower quality will therefore be discarded first.
  """


class EdcSamplePair(TypedDict):
  """The input data for one sample pair that is required to compute an EDC curve."""
  samples: tuple[EdcSample, EdcSample]
  """Data for the individual samples."""
  similarity_score: SimilarityScore
  """The similarity score for the sample pair."""


class EdcErrorType(Enum):
  """An error type for an EDC plot. The error is typically plotted on the Y-axis.
  This code currently supports EDC plots using the FNMR ("FNM-EDC") or the FMR ("FM-EDC").
  """
  FNMR = 'FNMR'
  """The FNMR (False Non-Match Rate),
  see https://www.iso.org/obp/ui/#iso:std:iso-iec:2382:-37:ed-3:v1:en:term:37.09.11.
  """
  FMR = 'FMR'
  """The FMR (False Match Rate),
  see <https://www.iso.org/obp/ui/#iso:std:iso-iec:2382:-37:ed-3:v1:en:term:37.09.09>.
  """


PairQualityScoreFunction = Callable[[QualityScore, QualityScore], QualityScore]
"""A function that combines the two quality scores of the `EdcSamplePair.samples` into a pairwise quality score.

Typically this should be the `min` function, since this will correspond to discarding samples (and their sample pairs)
in the ascending order of their individual quality scores.
"""


class _EdcInputBase(TypedDict):
  """Input required to compute an EDC curve.
  Note that this is just a base class for `_EdcInputSamplePair` and `_EdcInputNumpy`,
  either of which specify the complete input information that is required.
  """
  error_type: EdcErrorType
  """The error type of the EDC curve."""
  similarity_score_threshold: SimilarityScore
  """The threshold used to decide which comparisons are errors or not, which also depends on the error type."""
  pair_quality_score_function: Optional[PairQualityScoreFunction]
  """The `PairQualityScoreFunction`.
  If this isn't specified, it will be `min`, which is typically the correct function to use
  (as noted in the `PairQualityScoreFunction` description).
  """


class _EdcInputSamplePair(_EdcInputBase):
  """Input required to compute an EDC curve,
  in the form of a `EdcSamplePair` list in addition to the `_EdcInputBase` data.
  """
  sample_pairs: list[EdcSamplePair]


class _EdcInputNumpy(_EdcInputBase):
  """Input required to compute an EDC curve,
  in the form of two numpy arrays (`similarity_scores` and `pair_quality_scores`)
  in addition to the `_EdcInputBase` data.
  """
  similarity_scores: np.ndarray
  """A numpy array containing similarity scores.
  Each index corresponds to a `EdcSamplePair` and the `pair_quality_scores` value at the same index.
  """
  pair_quality_scores: np.ndarray
  """A numpy array containing pairwise quality scores (produced by a `PairQualityScoreFunction`).
  Each index corresponds to a `EdcSamplePair` and the `similarity_scores` value at the same index.

  These pairwise quality scores must be sorted in ascending order.
  """


_EdcInput = Union[_EdcInputSamplePair, _EdcInputNumpy]
"""Input required to compute an EDC curve.
Either a `_EdcInputSamplePair` or a `_EdcInputNumpy`.
"""


class EdcOutput(TypedDict):
  """The output of `compute_edc` that represents a computed EDC curve.

  Only the `EdcOutput.error_type`, `EdcOutput.error_fractions`, and
  the `EdcOutput.discard_fractions` are required to plot the EDC curve.
  """
  error_type: EdcErrorType
  """The error type of the EDC curve."""
  error_fractions: np.ndarray
  """The error fraction values of the EDC curve.
  Typically plotted on the Y-axis.
  At each index the value corresponds to the `discard_fractions` value at the same index.
  """
  discard_fractions: np.ndarray
  """The comparison discard fraction values of the EDC curve.
  Typically plotted on the X-axis.
  At each index the value corresponds to the `error_fractions` value at the same index.
  """
  error_counts: np.ndarray
  """The discrete integer counts of remaining errors.
  This is used to compute the `error_fractions`.
  At each index the value corresponds to the `discard_counts` value at the same index.
  """
  discard_counts: np.ndarray
  """The discrete integer comparison discard counts.
  This is used to compute the `discard_fractions`.
  At each index the value corresponds to the `error_counts` value at the same index.
  """
  comparison_count: int
  """The total number of comparisons."""


def compute_edc(
    error_type: EdcErrorType,
    sample_pairs: list[EdcSamplePair],
    similarity_score_threshold: Optional[SimilarityScore] = None,
    similarity_score_quantile: Optional[float] = None,
    starting_error: Optional[float] = None,
    pair_quality_score_function: PairQualityScoreFunction = min,
) -> EdcOutput:
  """Computes an EDC curve.

  Parameters
  ----------
  error_type : EdcErrorType
    The error type of the EDC curve.
  sample_pairs : list[EdcSamplePair]
    An `EdcSamplePair` list used to compute the EDC curve.
    This specifies the similarity scores and the quality scores.
  similarity_score_threshold : Optional[`SimilarityScore`]
    The threshold used to decide which comparisons are errors or not, which also depends on the error type.
    The parameters `similarity_score_quantile` or `starting_error` can be used instead.
  similarity_score_quantile : Optional[float]
    If set, the `similarity_score_threshold` will be computed as
    `np.quantile(similarity_scores, similarity_score_quantile)`.
    The parameters `similarity_score_threshold` or `starting_error` can be used instead.
  starting_error : Optional[float]
    If set, the `similarity_score_threshold` will be computed
    as `np.quantile(similarity_scores, starting_error)` for the FNMR error type,
    or as `np.quantile(similarity_scores, 1 - starting_error)` for the FMR error type.
    It is called "starting error" because the actual starting error at the 0% discard fraction will approximate
    this value, depending on the given similarity score distribution.
    The parameters `similarity_score_threshold` or `similarity_score_quantile` can be used instead.
  pair_quality_score_function : `PairQualityScoreFunction`
    The function used to get pairwise quality scores for each each `EdcSamplePair`
    from the two samples' individual quality scores.
    If this isn't specified, it will be `min`, which is typically the correct function to use
    (see the `PairQualityScoreFunction` description).

  Returns
  -------
  EdcOutput
    The data for the computed EDC curve.
  """
  edc_input = _EdcInputSamplePair(
      error_type=error_type,
      similarity_score_threshold=similarity_score_threshold,
      sample_pairs=sample_pairs,
      pair_quality_score_function=pair_quality_score_function,
  )
  edc_input_numpy = _get_edc_input_numpy(edc_input)

  assert sum(
      (0 if value is None else 1
       for value in (similarity_score_threshold, similarity_score_quantile, starting_error))) == 1, (
           'Excactly one of the parameters similarity_score_threshold, similarity_score_quantile, or starting_error'
           ' has to be specified.')
  if starting_error is not None:
    similarity_score_quantile = starting_error
    if error_type == EdcErrorType.FMR:
      similarity_score_quantile = 1 - similarity_score_quantile
  if similarity_score_threshold is None:
    similarity_score_threshold = np.quantile(edc_input_numpy['similarity_scores'], similarity_score_quantile)
    edc_input_numpy['similarity_score_threshold'] = similarity_score_threshold

  return _compute_edc(
      error_type=edc_input_numpy['error_type'],
      pair_quality_scores=edc_input_numpy['pair_quality_scores'],
      similarity_scores=edc_input_numpy['similarity_scores'],
      similarity_score_threshold=edc_input_numpy['similarity_score_threshold'],
  )


def _get_edc_input_numpy(edc_input: _EdcInput) -> _EdcInputNumpy:
  if ('pair_quality_scores' in edc_input) or ('pair_similarity_scores' in edc_input):
    assert 'sample_pairs' not in edc_input
    assert ('pair_quality_scores' in edc_input) and ('pair_similarity_scores' in edc_input)
    return edc_input
  else:
    assert 'sample_pairs' in edc_input
    sample_pairs = edc_input['sample_pairs']

    similarity_scores = np.zeros(len(sample_pairs), dtype=np.float64)
    pair_quality_scores = np.zeros(len(sample_pairs), dtype=np.float64)
    pair_quality_score_function = edc_input.get('pair_quality_score_function', min)
    for i, sample_pair in enumerate(sample_pairs):
      similarity_scores[i] = sample_pair['similarity_score']
      sample1, sample2 = sample_pair['samples']
      pair_quality_scores[i] = pair_quality_score_function(
          sample1['quality_score'],
          sample2['quality_score'],
      )

    order_scores = np.argsort(pair_quality_scores)
    pair_quality_scores = pair_quality_scores[order_scores]
    similarity_scores = similarity_scores[order_scores]

    edc_input_numpy: _EdcInputNumpy = {key: value for key, value in edc_input.items() if key != 'sample_pairs'}
    edc_input_numpy['similarity_scores'] = similarity_scores
    edc_input_numpy['pair_quality_scores'] = pair_quality_scores

    return edc_input_numpy


def _form_error_comparison_decision(error_type: EdcErrorType,
                                    similarity_score_or_scores: SimilarityScores,
                                    similarity_score_threshold: SimilarityScore,
                                    out: Optional[np.ndarray] = None):
  if error_type == EdcErrorType.FNMR:
    # FNMR, so non-matches are errors
    return np.less(similarity_score_or_scores, similarity_score_threshold, out=out)
  elif error_type == EdcErrorType.FMR:
    # FMR, so matches are errors
    return np.greater_equal(similarity_score_or_scores, similarity_score_threshold, out=out)


def _compute_edc(
    error_type: EdcErrorType,
    pair_quality_scores: np.ndarray,
    similarity_scores: np.ndarray,
    similarity_score_threshold: SimilarityScore,
) -> EdcOutput:
  """This contains the actual EDC computation.
  The `similarity_scores` are linked to the `pair_quality_scores`, and the scores must be sorted by the quality scores.
  """
  assert len(pair_quality_scores) == len(similarity_scores), "Input quality/comparison score count mismatch"

  # The array indices correspond to the discard counts, so 0 comparisons are discarded at index 0.
  comparison_count = len(pair_quality_scores)

  # Compute the (binary) per-comparison errors by comparing the comparison scores against the comparison_threshold:
  error_counts = np.zeros(comparison_count, dtype=np.uint32)
  _form_error_comparison_decision(error_type, similarity_scores, similarity_score_threshold, out=error_counts)
  # Then compute the cumulative error_counts sum:
  # The total error count will be at index 0, which corresponds to 0 discarded comparisons (or samples).
  # Conversely, at index comparison_count-1 only one comparison isn't discarded and the error count remains 0 or 1.
  error_counts = np.flipud(np.cumsum(np.flipud(error_counts), out=error_counts))

  # Usually the EDC should model the effect of discarding samples (instead of individual comparisons) based on
  # a progressively increasing quality threshold. This means that sequences of identical quality scores have to be
  # skipped at once. In this implementation the discard counts are equivalent to the array indices, so computing
  # the relevant array indices for the quality sequence starting points also obtains the corresponding discard counts:
  discard_counts = np.where(pair_quality_scores[:-1] != pair_quality_scores[1:])[0] + 1
  discard_counts = np.concatenate(([0], discard_counts))

  # Subtracting the discard_counts from the total comparison_count results in the remaining_counts:
  remaining_counts = comparison_count - discard_counts
  # Divide the relevant error_counts by the remaining_counts to compute the error_fractions
  error_fractions = error_counts[discard_counts] / remaining_counts
  # Divide the discard_counts by the total comparison_count to compute the discard_fractions:
  discard_fractions = discard_counts / comparison_count

  if discard_fractions[-1] != 1:  # NOTE Add a point at 100% discard for plotting edge cases.
    discard_fractions = np.concatenate((discard_fractions, [1]))
    error_fractions = np.concatenate((error_fractions, [0]))

  edc_output = EdcOutput(
      error_fractions=error_fractions,
      discard_fractions=discard_fractions,
      error_counts=error_counts,
      discard_counts=discard_counts,
      comparison_count=comparison_count,
  )
  return edc_output


def compute_edc_pauc(edc_output: EdcOutput, discard_fraction_limit: float) -> float:
  """This computes the pAUC value for the given EDC curve (with stepwise interpolation).

  Note that this does not automatically subtract the "area under theoretical best" value,
  as done in the paper "Finger image quality assessment features - definitions and evaluation".
  You can get that value by calling `compute_edc_area_under_theoretical_best` with the same parameters,
  and subtract the result thereof from the pAUC value result of this function
  (`compute_edc_pauc(edc_output, discard_fraction_limit) -
  compute_edc_area_under_theoretical_best(edc_output, discard_fraction_limit)`).

  Parameters
  ----------
  edc_output : EdcOutput
    The EDC curve data as returned by the `compute_edc` function.
    The required parts are the `EdcOutput.error_fractions` and the `EdcOutput.discard_fractions`.
  discard_fraction_limit : float
    The pAUC value for that discard fraction limit will be computed.
    I.e. if this is 1, the full AUC value is computed.
    Must be in [0,1].

  Returns
  -------
  float
    The computed pAUC value.
  """
  error_fractions, discard_fractions = edc_output['error_fractions'], edc_output['discard_fractions']
  assert len(error_fractions) == len(discard_fractions), 'error_fractions/discard_fractions length mismatch'
  assert discard_fraction_limit >= 0 and discard_fraction_limit <= 1, 'Invalid discard_fraction_limit'
  if discard_fraction_limit == 0:
    return 0
  pauc = 0
  for i in range(len(discard_fractions)):  # pylint: disable=consider-using-enumerate
    if i == (len(discard_fractions) - 1) or discard_fractions[i + 1] >= discard_fraction_limit:
      pauc += error_fractions[i] * (discard_fraction_limit - discard_fractions[i])
      break
    else:
      pauc += error_fractions[i] * (discard_fractions[i + 1] - discard_fractions[i])
  return pauc


def compute_edc_area_under_theoretical_best(edc_output: EdcOutput, discard_fraction_limit: float) -> float:
  """Computes the "area under theoretical best" up to the specified `discard_fraction_limit`.
  I.e. the area under the line defined by max(0, error_at_0%_discard minus `discard_fraction_limit`),
  the "error_at_0%_discard" being the `EdcOutput.error_fractions` value at index 0.

  This is commonly subtracted from the `compute_edc_pauc` value returned for the same parameters.

  Parameters
  ----------
  edc_output : EdcOutput
    The EDC curve data as returned by the `compute_edc` function.
    The only required part is the `EdcOutput.error_fractions` array.
  discard_fraction_limit : float
    The discard fraction cutoff point for the area.

  Returns
  -------
  float
    The computed "area under theoretical best" value.
  """
  starting_error = edc_output['error_fractions'][0]
  area = (starting_error**2) / 2
  if discard_fraction_limit < starting_error:
    area -= ((starting_error - discard_fraction_limit)**2) / 2
  return area
