from typing import Optional, Dict, Union, List, Tuple
import torch

def patch_beam_scorer(BeamSearchScorer):
    """
    Patch beam scorer to include prompt length in the length penalty
    """
    if hasattr(BeamSearchScorer, "patched"):
        return

    print("Patching beam scorer")
    BeamSearchScorer.patched = True
    beam_score_process_og = BeamSearchScorer.process
    beam_score_finalize_og = BeamSearchScorer.finalize

    def beam_score_process_override(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        group_index: Optional[int] = 0,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        return beam_score_process_og(self, input_ids, next_scores, next_tokens, next_indices, pad_token_id,
                                    eos_token_id, beam_indices, group_index, decoder_prompt_len=0)

    def beam_score_finalize_override(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Tuple[torch.LongTensor]:
        return beam_score_finalize_og(self, input_ids, final_beam_scores, final_beam_tokens, final_beam_indices,
                                    max_length, pad_token_id, eos_token_id, beam_indices, decoder_prompt_len=0)

    BeamSearchScorer.process = beam_score_process_override
    BeamSearchScorer.finalize = beam_score_finalize_override
