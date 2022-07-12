# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""
from sklearn.metrics import accuracy_score

from moses.metrics import fraction_valid

import evaluate
import datasets


# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:module,
title = {Molecular Validity Assessment},
authors={daiyizheng},
year={2022}
}
"""


_DESCRIPTION = """\
Valid is the proportion of correct molecules in the total number of molecules. It can be computed with:
valid_rate = valid/total
"""


_KWARGS_DESCRIPTION = """
Args:
    similes (`list` of `string`): generate molecules.
Returns:
    valid: valid score, Minimum possible value is 0. Maximum possible value is 1.0.
Examples:
    Examples should be written in doctest format, and should illustrate how
        >>> accuracy_metric = evaluate.load("valid")
        >>> results = accuracy_metric.compute(similes=["CC1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1","CC1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1"])
        >>> print(results)
        {'valid': 0.98}
"""


BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Valid(evaluate.Metric):
    """TODO: Short description of my evaluation module."""

    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                    "smiles": datasets.Value("string", id="sequence")
            }),
            # Homepage of the module for documentation
            homepage="http://module.homepage",
            # Additional links to the codebase or references
            codebase_urls=[],
            reference_urls=[]
        )

    # def _download_and_prepare(self, dl_manager):
    #     """Optional: download external resources useful to compute the scores"""
    #
    #     pass

    def _compute(self, smiles, n_jobs):
        return {
            "valid": fraction_valid(smiles, n_jobs=n_jobs)
        }

    # def add(self, predictions):
    #     pass
    #
    # def add_batch(self, *, predictions=None, references=None, **kwargs):
    #     pass