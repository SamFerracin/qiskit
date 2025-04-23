// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use num_complex::Complex64;
// The lookup tables are pretty illegible if we have all the syntactic noise of `PauliBitTerm`.
use super::PauliBitTerm::{self, *};

/// Short-hand alias for [Complex64::new] that retains its ability to be used in `const` contexts.
const fn c64(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// The allowable items of `PauliBitTerm`. This is used by the lookup expansion; we need a const-safe
/// way of iterating through all of the variants.
static PAULI_BIT_TERMS: [PauliBitTerm; 3] = [X, Y, Z];