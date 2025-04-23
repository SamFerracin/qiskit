// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

mod lookup;

use num_complex::Complex64;
use thiserror::Error;

/// Named handle to the alphabet of single-qubit Pauli terms.
///
/// This is just the Rust-space representation.  We make a separate Python-space `enum.IntEnum` to
/// represent the same information, since we enforce strongly typed interactions in Rust, including
/// not allowing the stored values to be outside the valid `PauliBitTerm`s, but doing so in Python
/// would make it very difficult to use the class efficiently with Numpy array views.  We attach this
/// sister class of `PauliBitTerm` to `NoiseMap` as a scoped class.
///
/// # Representation
///
/// The `u8` representation and the exact numerical values of these are part of the public API. The
/// two bits store the symplectic Pauli representation with Z in the Lsb0 and X in the Lsb1, that is,
/// `0b01` for `Z`, `0b10` for `X`, and `0b11` for `Y`.
///
/// The `0b00` representation thus ends up being the natural representation of the `I` operator,
/// but this is never stored, and is not named in the enumeration.
///
/// This operator does not store phase terms of $-i$. Any additional phase is stored only in a
/// corresponding coefficient.
///
/// # Dev notes
///
/// This type is required to be `u8`, but it's a subtype of `u8` because not all `u8` are valid
/// `PauliBitTerm`s.  For interop with Python space, we accept Numpy arrays of `u8` to represent this,
/// which we transmute into slices of `PauliBitTerm`, after checking that all the values are correct (or
/// skipping the check if Python space promises that it upheld the checks).
///
/// We deliberately _don't_ impl `numpy::Element` for `PauliBitTerm` (which would let us accept and
/// return `PyArray1<PauliBitTerm>` at Python-space boundaries) so that it's clear when we're doing
/// the transmute, and we have to be explicit about the safety of that.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum PauliBitTerm {
    /// Pauli X operator.
    X = 0b10,
    /// Pauli Y operator.
    Y = 0b11,
    /// Pauli Z operator.
    Z = 0b01,
}
impl From<PauliBitTerm> for u8 {
    fn from(value: PauliBitTerm) -> u8 {
        value as u8
    }
}
unsafe impl ::bytemuck::CheckedBitPattern for PauliBitTerm {
    type Bits = u8;

    // TODO: Check this
    #[inline(always)]
    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits <= 0b11_11 && (*bits & 0b11_00) < 0b11_00 && (*bits & 0b00_11) != 0
    }
}
unsafe impl ::bytemuck::NoUninit for PauliBitTerm {}

impl PauliBitTerm {
    /// Get the label of this `PauliBitTerm` used in Python-space applications.  This is a single-letter
    /// string.
    #[inline]
    pub fn py_label(&self) -> &'static str {
        // Note: these labels are part of the stable Python API and should not be changed.
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }

    /// Get the name of this `PauliBitTerm`, which is how Python space refers to the integer constant.
    #[inline]
    pub fn py_name(&self) -> &'static str {
        // Note: these names are part of the stable Python API and should not be changed.
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }

    /// Attempt to convert a `u8` into `PauliBitTerm`.
    ///
    /// Unlike the implementation of `TryFrom<u8>`, this allows `b'I'` as an alphabet letter,
    /// returning `Ok(None)` for it.  All other letters outside the alphabet return the complete
    /// error condition.
    #[inline]
    fn try_from_u8(value: u8) -> Result<Option<Self>, BitTermFromU8Error> {
        match value {
            b'I' => Ok(None),
            b'X' => Ok(Some(PauliBitTerm::X)),
            b'Y' => Ok(Some(PauliBitTerm::Y)),
            b'Z' => Ok(Some(PauliBitTerm::Z)),
            _ => Err(BitTermFromU8Error(value)),
        }
    }

    /// Does this term include an X component in its ZX representation?
    ///
    /// This is true for the operators and eigenspace projectors associated with X and Y.
    pub fn has_x_component(&self) -> bool {
        ((*self as u8) & (Self::X as u8)) != 0
    }

    /// Does this term include a Z component in its ZX representation?
    ///
    /// This is true for the operators and eigenspace projectors associated with Y and Z.
    pub fn has_z_component(&self) -> bool {
        ((*self as u8) & (Self::Z as u8)) != 0
    }
}

fn bit_term_as_pauli(bit: &PauliBitTerm) -> &'static [(bool, Option<PauliBitTerm>)] {
    match bit {
        PauliBitTerm::X => &[(true, Some(PauliBitTerm::X))],
        PauliBitTerm::Y => &[(true, Some(PauliBitTerm::Y))],
        PauliBitTerm::Z => &[(true, Some(PauliBitTerm::Z))],
    }
}

/// The error type for a failed conversion into `PauliBitTerm`.
#[derive(Error, Debug)]
#[error("{0} is not a valid letter of the single-qubit alphabet")]
pub struct BitTermFromU8Error(u8);

// `PauliBitTerm` allows safe `as` casting into `u8`.  This is the reverse, which is fallible,
// because `PauliBitTerm` is a value-wise subtype of `u8`.
impl ::std::convert::TryFrom<u8> for PauliBitTerm {
    type Error = BitTermFromU8Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        ::bytemuck::checked::try_cast(value).map_err(|_| BitTermFromU8Error(value))
    }
}

/// Error cases stemming from data coherence at the point of entry into `SparseObservable` from
/// user-provided arrays.
///
/// These most typically appear during [from_raw_parts], but can also be introduced by various
/// remapping arithmetic functions.
///
/// These are generally associated with the Python-space `ValueError` because all of the
/// `TypeError`-related ones are statically forbidden (within Rust) by the language, and conversion
/// failures on entry to Rust from Python space will automatically raise `TypeError`.
#[derive(Error, Debug)]
pub enum CoherenceError {
    #[error("`boundaries` ({boundaries}) must be one element longer than `coeffs` ({coeffs})")]
    MismatchedTermCount { coeffs: usize, boundaries: usize },
    #[error("`bit_terms` ({bit_terms}) and `indices` ({indices}) must be the same length")]
    MismatchedItemCount { bit_terms: usize, indices: usize },
    #[error("the first item of `boundaries` ({0}) must be 0")]
    BadInitialBoundary(usize),
    #[error("the last item of `boundaries` ({last}) must match the length of `bit_terms` and `indices` ({items})")]
    BadFinalBoundary { last: usize, items: usize },
    #[error("all qubit indices must be less than the number of qubits")]
    BitIndexTooHigh,
    #[error("the values in `boundaries` include backwards slices")]
    DecreasingBoundaries,
    #[error("the values in `indices` are not term-wise increasing")]
    UnsortedIndices,
    #[error("the input contains duplicate qubits")]
    DuplicateIndices,
    #[error("the provided qubit mapping does not account for all contained qubits")]
    IndexMapTooSmall,
    #[error("cannot shrink the qubit count in an observable from {current} to {target}")]
    NotEnoughQubits { current: usize, target: usize },
}

/// A container that stores noise maps in a qubit-sparse format.
///
/// See [PyNoiseMap] for detailed docs.
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseMap {
    /// The number of qubits the noise map is defined for.  This is not inferable from any other
    /// shape or values, since identities are not stored explicitly.
    num_qubits: u32,
    /// The coefficients of each Pauli term. This has as many elements as `bit_terms`.
    coeffs: Vec<Complex64>,
    /// A flat list of single-qubit Pauli terms.  This is more naturally a list of lists, but is stored
    /// flat for memory usage and locality reasons, with the sublists denoted by `boundaries.`
    bit_terms: Vec<PauliBitTerm>,
    /// A flat list of the qubit indices that the corresponding entries in `bit_terms` act on.  This
    /// list must always be term-wise sorted, where a term is a sublist as denoted by `boundaries`.
    indices: Vec<u32>,
    /// Indices that partition `bit_terms` and `indices` into sublists for each individual term in
    /// the sum.  `boundaries[0]..boundaries[1]` is the range of indices into `bit_terms` and
    /// `indices` that correspond to the first term of the sum.  All unspecified qubit indices are
    /// implicitly the identity.  This is one item longer than `coeffs`, since `boundaries[0]` is
    /// always an explicit zero (for algorithmic ease).
    boundaries: Vec<usize>,
}

impl NoiseMap {
    /// Create a noise map from the raw components that make it up.
    ///
    /// This checks the input values for data coherence on entry.
    pub fn new(
        num_qubits: u32,
        coeffs: Vec<Complex64>,
        bit_terms: Vec<PauliBitTerm>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Result<Self, CoherenceError> {
        if coeffs.len() + 1 != boundaries.len() {
            return Err(CoherenceError::MismatchedTermCount {
                coeffs: coeffs.len(),
                boundaries: boundaries.len(),
            });
        }
        if bit_terms.len() != indices.len() {
            return Err(CoherenceError::MismatchedItemCount {
                bit_terms: bit_terms.len(),
                indices: indices.len(),
            });
        }
        // We already checked that `boundaries` is at least length 1.
        if *boundaries.first().unwrap() != 0 {
            return Err(CoherenceError::BadInitialBoundary(boundaries[0]));
        }
        if *boundaries.last().unwrap() != indices.len() {
            return Err(CoherenceError::BadFinalBoundary {
                last: *boundaries.last().unwrap(),
                items: indices.len(),
            });
        }
        for (&left, &right) in boundaries[..].iter().zip(&boundaries[1..]) {
            if right < left {
                return Err(CoherenceError::DecreasingBoundaries);
            }
            let indices = &indices[left..right];
            if !indices.is_empty() {
                for (index_left, index_right) in indices[..].iter().zip(&indices[1..]) {
                    if index_left == index_right {
                        return Err(CoherenceError::DuplicateIndices);
                    } else if index_left > index_right {
                        return Err(CoherenceError::UnsortedIndices);
                    }
                }
            }
            if indices.last().map(|&ix| ix >= num_qubits).unwrap_or(false) {
                return Err(CoherenceError::BitIndexTooHigh);
            }
        }
        // SAFETY: we've just done the coherence checks.
        Ok(unsafe { Self::new_unchecked(num_qubits, coeffs, bit_terms, indices, boundaries) })
    }

    /// Create a new noise map from the raw components without checking data coherence.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure that the data-coherence requirements, as enumerated in the
    /// struct-level documentation, have been upheld.
    #[inline(always)]
    pub unsafe fn new_unchecked(
        num_qubits: u32,
        coeffs: Vec<Complex64>,
        bit_terms: Vec<PauliBitTerm>,
        indices: Vec<u32>,
        boundaries: Vec<usize>,
    ) -> Self {
        Self {
            num_qubits,
            coeffs,
            bit_terms,
            indices,
            boundaries,
        }
    }

    /// Get the number of qubits the noise map is defined on.
    #[inline]
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Get the number of terms in the noise map.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.coeffs.len()
    }
}