import os
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from src.dreamer_scratch.train import build, signature, check_checkpoint, AllocationTimer, Progress, Bound


class ScratchTests(unittest.TestCase):
    def test_dictionary_metrics_default_to_distributed_reduction(self):
        class Parent:
            def log_dict(self, dictionary, **kwargs):return kwargs
        class Model(Bound, Parent):pass
        self.assertTrue(Model().log_dict({'loss':1.})['sync_dist'])

    def test_resumed_allocation_gets_fresh_time_budget(self):
        timer=AllocationTimer(duration={'hours':47},interval='step')
        initial=timer.state_dict()
        timer.load_state_dict({'time_elapsed':{'train':47*3600,'validate':1,'test':0}})
        self.assertEqual(timer.state_dict(),initial)

    def test_nonfinite_training_stops(self):
        callback=Progress()
        callback.on_train_batch_end(None,None,{'loss':torch.tensor(1.)},None,0)
        with self.assertRaises(FloatingPointError):
            callback.on_train_batch_end(None,None,{'loss':torch.tensor(float('nan'))},None,0)

    def test_no_pretrained_load_and_reproducible_initialization(self):
        upstream=Path(os.environ['OSU_PROJECT_ROOT'])/'upstream/osu-dreamer'
        for stage in ('codec','generator','style'):
            with self.subTest(stage=stage), patch('torch.load',side_effect=AssertionError('pretrained read')):
                torch.manual_seed(1731);model=build(stage,upstream);a=signature(model);del model
                torch.manual_seed(1731);model=build(stage,upstream);b=signature(model);del model
                torch.manual_seed(1732);model=build(stage,upstream);c=signature(model);del model
                self.assertEqual(a,b);self.assertNotEqual(a,c)

    def test_reject_foreign_checkpoint(self):
        with patch('torch.load',return_value={'state_dict':{}}):
            with self.assertRaisesRegex(ValueError,'not from this scratch'):
                check_checkpoint('not-read.pt',{'schema':'scratch'},'codec')

    def test_reject_wrong_stage(self):
        binding={'schema':'scratch'}
        with patch('torch.load',return_value={'scratch_binding':binding,'scratch_stage':'style'}):
            with self.assertRaises(ValueError):check_checkpoint('not-read.pt',binding,'codec')
