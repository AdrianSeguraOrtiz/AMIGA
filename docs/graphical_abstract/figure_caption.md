# Figure description

AMIGA learns to prioritize existing consensus gene regulatory networks.
Upstream multi-objective optimization supplies a Pareto front of weighted
combinations of base networks. Benchmark reference links provide candidate
quality, illustrated by AUPR. AMIGA represents each candidate using mixture
weights, objectives, reconstructed network descriptors and expression context.
Quality supervises learning and is excluded from prediction inputs.

Grouped validation keeps fronts intact. The reusable core groups by `front_id`;
the experimental workflow additionally groups related conditions by topology
and performs nested label, parameter and column selection. The graphical
overview depicts the common training/deployment principle, rather than all
experimental phases or a particular chosen backend.

After validation and final fitting, the saved ranker and predictor schema score
new candidates without a reference network. AMIGA exports scores and ranks
within each front, enabling a single recommendation or a shortlist. The
highlighted network is the same schematic candidate C, not a newly generated
graph. The matrices, fronts, networks and score bars are illustrative; predicted
scores are not measured AUPR or evidence of causal biological correctness.
