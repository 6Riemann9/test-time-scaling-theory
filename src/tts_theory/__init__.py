"""tts_theory: Test-Time Scaling 相变理论、双精度积分与抗坍塌算法包。"""

from __future__ import annotations

from tts_theory.algorithms import (
    BaseSelector,
    CalibratedWeightedBoN,
    EmpiricalScoreCalibrator,
    MajorityVoting,
    OptimalNStarBoN,
    StandardBoN,
    TwoStageVerifiedMajority,
)
from tts_theory.calibration_theory import (
    bernstein_zero_collapse_threshold,
    build_score_bins,
    calibration_minimax_scaling,
    compute_bin_calibrator_scores,
    exact_conditional_bin_bon_accuracy,
    joint_calibration_compute_scaling,
    simulate_finite_calibration_scaling,
)
from tts_theory.ensemble_theory import (
    GaussianCopulaHeavyTailSpec,
    GeneralGaussianMultiVerifierSpec,
    HeavyTailMultiVerifierSpec,
    HeterogeneousVulnerabilitySpec,
    MinEnsembleVerifierSpec,
    MultiVerifierGaussianSpec,
    ensemble_phase_reversal_criterion,
    gaussian_copula_heavy_tail_criterion,
    general_min_ensemble_qp_criterion,
    general_min_tail_asymptotic_constants,
    heavy_tail_separation_criterion,
    optimal_linear_ensemble_criterion,
    simulate_multi_verifier_bon,
    solve_positive_quadrant_qp,
)
from tts_theory.heterogeneity_theory import (
    PromptHeterogeneitySpec,
    diagnose_simpson_confounding,
    diagnose_within_prompt_residual,
    evaluate_pooled_vs_within_prompt_calibration,
)
from tts_theory.integrals import (
    exact_bon_accuracy,
    exact_bon_curve,
    exact_calibrated_bon_accuracy,
    numerical_optimal_n,
)
from tts_theory.real_eval import (
    diagnose_multi_verifier_correlations,
    diagnose_real_verifier_distribution,
    evaluate_discrete_algorithms_on_real_pools,
    evaluate_finite_calibration_on_real_pools,
    evaluate_heterogeneous_multi_rm_on_real_pools,
    evaluate_joint_m_n_calibration_on_real_pools,
    evaluate_real_prompt_heterogeneity,
    exact_pool_bon_accuracy,
    extract_boxed_answer,
    fit_and_predict_sota_calibrator,
    is_math_answer_correct,
    is_math_answer_correct_any,
    normalize_math_answer,
)
from tts_theory.simulator import (
    DiscreteAnswerTaskSpec,
    simulate_algorithms_comparison,
    simulate_binary_bon,
    simulate_bon,
)
from tts_theory.theorems import (
    BinaryVerifierSpec,
    ContaminatedVerifierSpec,
    ContinuousVerifierSpec,
    GaussianVerifierSpec,
    binary_asymptotic_ceiling,
    binary_bon_accuracy,
    binary_bon_divergence_bounds,
    binary_marginal_gain,
    binary_optimal_budget,
    calibrated_bon_asymptotic_limit,
    check_collapse_criterion,
    closed_form_optimal_n,
    gaussian_peak_threshold,
    weighted_majority_convergence_bound,
)

__all__ = [
    # 定理与分布规格
    "ContinuousVerifierSpec",
    "GaussianVerifierSpec",
    "ContaminatedVerifierSpec",
    "BinaryVerifierSpec",
    "check_collapse_criterion",
    "gaussian_peak_threshold",
    "closed_form_optimal_n",
    "binary_bon_accuracy",
    "binary_bon_divergence_bounds",
    "binary_asymptotic_ceiling",
    "binary_marginal_gain",
    "binary_optimal_budget",
    "calibrated_bon_asymptotic_limit",
    "weighted_majority_convergence_bound",
    # 双精度数值积分引擎
    "exact_bon_accuracy",
    "exact_bon_curve",
    "numerical_optimal_n",
    "exact_calibrated_bon_accuracy",
    # 蒙特卡洛仿真器
    "DiscreteAnswerTaskSpec",
    "simulate_bon",
    "simulate_binary_bon",
    "simulate_algorithms_comparison",
    # 推理选择与校准算法
    "BaseSelector",
    "StandardBoN",
    "OptimalNStarBoN",
    "EmpiricalScoreCalibrator",
    "CalibratedWeightedBoN",
    "TwoStageVerifiedMajority",
    "MajorityVoting",
    # V2 & V4/V5 多裁判相变逆转与重尾/Copula 统一理论（定理 5）
    "MultiVerifierGaussianSpec",
    "GeneralGaussianMultiVerifierSpec",
    "MinEnsembleVerifierSpec",
    "HeavyTailMultiVerifierSpec",
    "GaussianCopulaHeavyTailSpec",
    "HeterogeneousVulnerabilitySpec",
    "solve_positive_quadrant_qp",
    "general_min_tail_asymptotic_constants",
    "optimal_linear_ensemble_criterion",
    "general_min_ensemble_qp_criterion",
    "ensemble_phase_reversal_criterion",
    "heavy_tail_separation_criterion",
    "gaussian_copula_heavy_tail_criterion",
    "simulate_multi_verifier_bon",
    # V2 & V4/V5 有限校准集二次坍塌、KL-LCB 与联合标度（定理 6）
    "build_score_bins",
    "exact_conditional_bin_bon_accuracy",
    "compute_bin_calibrator_scores",
    "bernstein_zero_collapse_threshold",
    "calibration_minimax_scaling",
    "joint_calibration_compute_scaling",
    "simulate_finite_calibration_scaling",
    # V4/V5 题目异质性、辛普森悖论与题内标准化（定理 7）
    "PromptHeterogeneitySpec",
    "diagnose_simpson_confounding",
    "diagnose_within_prompt_residual",
    "evaluate_pooled_vs_within_prompt_calibration",
    # V3/Stage2 真实大模型候选池与开源 RM 评估接口
    "extract_boxed_answer",
    "normalize_math_answer",
    "is_math_answer_correct",
    "is_math_answer_correct_any",
    "exact_pool_bon_accuracy",
    "diagnose_real_verifier_distribution",
    "diagnose_multi_verifier_correlations",
    "evaluate_finite_calibration_on_real_pools",
    "evaluate_discrete_algorithms_on_real_pools",
    "fit_and_predict_sota_calibrator",
    "evaluate_heterogeneous_multi_rm_on_real_pools",
    "evaluate_joint_m_n_calibration_on_real_pools",
    "evaluate_real_prompt_heterogeneity",
]

__version__ = "0.5.0"
