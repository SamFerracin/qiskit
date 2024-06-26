# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module containing synthesis algorithms for Permutation gates."""


from .permutation_lnn import synth_permutation_depth_lnn_kms
from .permutation_full import synth_permutation_basic, synth_permutation_acg
from .permutation_reverse_lnn import synth_permutation_reverse_lnn_kms
