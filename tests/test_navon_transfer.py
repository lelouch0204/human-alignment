import sys
from pathlib import Path
import unittest
import numpy as np
from sklearn.linear_model import RidgeClassifier

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from human_alignment.navon_transfer import ridge_path_predict, inner_splits, transfer_probe, matched_gaps


class TransferTests(unittest.TestCase):
    def test_ridge_path_matches_sklearn(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(100,24)); valid=rng.normal(size=(30,24))
        y=np.asarray(list("ABCD")*25)
        alphas=(1e-6,0.01,1,10000)
        for a,p in zip(alphas,ridge_path_predict(x,y,valid,alphas)):
            expected=RidgeClassifier(alpha=a,solver="cholesky").fit(x,y).predict(valid)
            np.testing.assert_array_equal(p,expected)

    def test_inner_splits_hold_out_both_axes(self):
        groups=np.tile(np.repeat(list("ABCDEFGHIJ"),3),2)
        conditions=np.repeat(["s10","s14"],30)
        splits=list(inner_splits(np.arange(60),groups,conditions))
        self.assertEqual(len(splits),6)
        for fit,valid in splits:
            self.assertFalse(set(groups[fit])&set(groups[valid]))
            self.assertFalse(set(conditions[fit])&set(conditions[valid]))

    def test_ridge_path_matches_on_rank_deficient_features(self):
        rng=np.random.default_rng(9)
        x=rng.normal(size=(40,60)); valid=rng.normal(size=(20,60))
        x=np.column_stack((x,x[:,:20])); valid=np.column_stack((valid,valid[:,:20]))
        y=np.asarray(list("ABCD")*10)
        alphas=(1e-6,0.01,100)
        for a,p in zip(alphas,ridge_path_predict(x,y,valid,alphas)):
            expected=RidgeClassifier(alpha=a,solver="cholesky").fit(x,y).predict(valid)
            np.testing.assert_array_equal(p,expected)

    def test_separable_transfer_and_matched_controls(self):
        rows=[]; vectors=[]
        letters=list("ABCDEFGHIJ")
        for c in ("s10","s14","s18"):
            for g,G in enumerate(letters):
                for l,L in enumerate(letters):
                    if g==l: continue
                    rows.append(dict(image_id=f"{c}{G}{L}",global_letter=G,local_letter=L,condition=c,pos_idx=0))
                    vectors.append(np.r_[np.eye(10)[g],np.eye(10)[l]]/np.sqrt(2))
        for target in ("global","local"):
            pred,folds=transfer_probe(np.asarray(vectors),rows,target,alphas=(0.01,1))
            self.assertEqual(len(folds),15)
            self.assertEqual(len(pred),3*len(rows))
            self.assertEqual(sum(r['domain']=='out' for r in pred),len(rows))
            self.assertTrue(all(r['correct'] for r in pred))
            self.assertTrue(all(r['gap']==0 for r in matched_gaps(pred)))


if __name__=="__main__": unittest.main()
