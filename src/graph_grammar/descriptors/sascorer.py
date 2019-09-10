import gzip
import math
import os
import pickle
from rdkit import Chem
from rdkit.six import iteritems
from rdkit.Chem import rdMolDescriptors


def synthetic_accessibility(mol, _fscores=None):
    '''
    calculation of synthetic accessibility score as described in:

    'Estimation of Synthetic Accessibility Score of Drug-like Molecules 
    based on Molecular Complexity and Fragment Contributions'
    Peter Ertl and Ansgar Schuffenhauer
    Journal of Cheminformatics 1:8 (2009)
    http://www.jcheminf.com/content/1/1/8

    several small modifications to the original paper are included
    particularly slightly different formula for marocyclic penalty
    and taking into account also molecule symmetry (fingerprint density)

    for a set of 10k diverse molecules the agreement between the original method
    as implemented in PipelinePilot and this implementation is r2 = 0.97

    peter ertl & greg landrum, september 2013

    Parameters
    ----------
    mol : Mol

    Returns
    -------
    float : synthetic accessibility score
    '''
    if _fscores is None:
        with gzip.open(os.path.join(os.path.dirname(__file__), 'fpscores.pkl.gz'), 'rb') as f:
            _fscores = pickle.load(f)

    out_dict = {}
    for each_list in _fscores:
        for each_idx in range(1,len(each_list)):
            out_dict[each_list[each_idx]] = float(each_list[0])
    _fscores = out_dict

    # fragment score
    # 2 is the *radius* of the circular fingerprint
    fingerprint = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fingerprints = fingerprint.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bit_id, value in iteritems(fingerprints):
        nf += value
        sfp = bit_id
        score1 += _fscores.get(sfp, -4) * value
    score1 /= nf

    # features score
    num_atoms = mol.GetNumAtoms()
    num_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ring_info = mol.GetRingInfo()
    num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    num_bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    num_macrocycles = 0
    for each_ring in ring_info.AtomRings():
        if len(each_ring) > 8:
            num_macrocycles += 1

    size_penalty = num_atoms ** 1.005 - num_atoms
    stereo_penalty = math.log10(num_chiral_centers + 1)
    spiro_penalty = math.log10(num_spiro + 1)
    bridge_penalty = math.log10(num_bridgeheads + 1)
    macrocycle_penalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocycle_penalty = math.log10(num_macrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if num_macrocycles > 0:
        macrocycle_penalty = math.log10(2)

    score2 = 0. -size_penalty -stereo_penalty -spiro_penalty -bridge_penalty -macrocycle_penalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if num_atoms > len(fingerprints):
        score3 = math.log(float(num_atoms) / len(fingerprints)) * .5

    sascore = score1 + score2 + score3
    
    # need to transform "raw" value into scale between 1 and 10
    min_score = -4.0
    max_score = 2.5
    sascore = 11. - (sascore - min_score + 1) / (max_score - min_score) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore+1.-9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore


def synthetic_accessibility_batch(mol_list, print_freq=10000, logger=print, _fscores=None):
    '''
    calculation of synthetic accessibility score as described in:

    'Estimation of Synthetic Accessibility Score of Drug-like Molecules 
    based on Molecular Complexity and Fragment Contributions'
    Peter Ertl and Ansgar Schuffenhauer
    Journal of Cheminformatics 1:8 (2009)
    http://www.jcheminf.com/content/1/1/8

    several small modifications to the original paper are included
    particularly slightly different formula for marocyclic penalty
    and taking into account also molecule symmetry (fingerprint density)

    for a set of 10k diverse molecules the agreement between the original method
    as implemented in PipelinePilot and this implementation is r2 = 0.97

    peter ertl & greg landrum, september 2013

    Parameters
    ----------
    mol_list : list of Mol

    Returns
    -------
    list of floats : synthetic accessibility score
    '''
    if _fscores is None:
        with gzip.open(os.path.join(os.path.dirname(__file__), 'fpscores.pkl.gz'), 'rb') as f:
            _fscores = pickle.load(f)

    out_dict = {}
    for each_list in _fscores:
        for each_idx in range(1,len(each_list)):
            out_dict[each_list[each_idx]] = float(each_list[0])
    _fscores = out_dict

    sascore_list = []
    for each_idx, each_mol in enumerate(mol_list):
        if each_mol is None:
            sascore_list.append(None)
            continue

        if each_idx % print_freq == 0 and each_idx != 0:
            logger(f'{each_idx} completed')
        # fragment score
        # 2 is the *radius* of the circular fingerprint
        fingerprint = rdMolDescriptors.GetMorganFingerprint(each_mol, 2)
        fingerprints = fingerprint.GetNonzeroElements()
        score1 = 0.
        nf = 0
        for bit_id, value in iteritems(fingerprints):
            nf += value
            sfp = bit_id
            score1 += _fscores.get(sfp, -4) * value
        score1 /= nf

        # features score
        num_atoms = each_mol.GetNumAtoms()
        num_chiral_centers = len(Chem.FindMolChiralCenters(each_mol, includeUnassigned=True))
        ring_info = each_mol.GetRingInfo()
        num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(each_mol)
        num_bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(each_mol)
        num_macrocycles = 0
        for each_ring in ring_info.AtomRings():
            if len(each_ring) > 8:
                num_macrocycles += 1

        size_penalty = num_atoms ** 1.005 - num_atoms
        stereo_penalty = math.log10(num_chiral_centers + 1)
        spiro_penalty = math.log10(num_spiro + 1)
        bridge_penalty = math.log10(num_bridgeheads + 1)
        macrocycle_penalty = 0.
        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocycle_penalty = math.log10(num_macrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if num_macrocycles > 0:
            macrocycle_penalty = math.log10(2)

        score2 = 0. -size_penalty -stereo_penalty -spiro_penalty -bridge_penalty -macrocycle_penalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if num_atoms > len(fingerprints):
            score3 = math.log(float(num_atoms) / len(fingerprints)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min_score = -4.0
        max_score = 2.5
        sascore = 11. - (sascore - min_score + 1) / (max_score - min_score) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore+1.-9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0
        sascore_list.append(sascore)
    return sascore_list

#
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
#
#     * Redistributions of source code must retain the above copyright 
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following 
#       disclaimer in the documentation and/or other materials provided 
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc. 
#       nor the names of its contributors may be used to endorse or promote 
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
