"""The locked evaluation.  These pin the semantics the protocol states."""

import numpy as np
import pytest

from polyptail.eval.metrics import (
    aggregate, binary_scores, confusion, hd95, image_metrics, sweep_scores,
)


def box(shape, y0, y1, x0, x1):
    a = np.zeros(shape, dtype=bool)
    a[y0:y1, x0:x1] = True
    return a


class TestBinaryScores:
    def test_known_overlap(self):
        gt = box((64, 64), 20, 40, 20, 40)
        pred = box((64, 64), 22, 42, 22, 42)
        s = binary_scores(pred, gt)
        inter = 18 * 18
        assert s["dice"] == pytest.approx(2 * inter / (400 + 400))
        assert s["iou"] == pytest.approx(inter / (800 - inter))
        assert s["recall"] == pytest.approx(inter / 400)
        assert s["precision"] == pytest.approx(inter / 400)

    def test_perfect_and_disjoint(self):
        gt = box((32, 32), 4, 12, 4, 12)
        assert binary_scores(gt, gt)["dice"] == 1.0
        assert binary_scores(box((32, 32), 20, 28, 20, 28), gt)["dice"] == 0.0

    def test_both_empty_is_a_correct_answer(self):
        z = np.zeros((16, 16), bool)
        s = binary_scores(z, z)
        assert s["dice"] == 1.0 and s["iou"] == 1.0
        assert np.isnan(s["recall"]) and np.isnan(s["precision"])

    def test_empty_prediction_scores_zero_not_epsilon(self):
        """No smoothing: a total failure must read as a total failure, or the
        left tail this project is about gets quietly lifted."""
        gt = box((32, 32), 4, 12, 4, 12)
        s = binary_scores(np.zeros((32, 32), bool), gt)
        assert s["dice"] == 0.0 and s["iou"] == 0.0 and s["recall"] == 0.0
        assert np.isnan(s["precision"]) and s["pred_empty"]

    def test_iou_never_exceeds_dice(self):
        """The invariant that several widely-copied published tables violate
        (brief §1.3 W6)."""
        rng = np.random.default_rng(0)
        for _ in range(3000):
            a = rng.random((12, 12)) > rng.uniform(0.1, 0.9)
            b = rng.random((12, 12)) > rng.uniform(0.1, 0.9)
            s = binary_scores(a, b)
            assert s["iou"] <= s["dice"] + 1e-12

    def test_confusion_partitions_the_image(self):
        rng = np.random.default_rng(1)
        a, b = rng.random((20, 20)) > 0.5, rng.random((20, 20)) > 0.5
        tp, fp, fn, tn = confusion(a, b)
        assert tp + fp + fn + tn == a.size


class TestHD95:
    def test_identical_masks_have_zero_distance(self):
        gt = box((64, 64), 10, 30, 10, 30)
        assert hd95(gt, gt) == pytest.approx(0.0)

    def test_one_pixel_shift(self):
        gt = box((64, 64), 10, 30, 10, 30)
        pred = box((64, 64), 11, 31, 10, 30)
        assert 0.0 < hd95(pred, gt) <= 2.0

    def test_empty_prediction_returns_the_image_diagonal_not_nan(self):
        """Dropping undefined cases would make HD95 optimistic on exactly the
        failures that matter, so it is reported at the grid's largest distance
        and counted separately."""
        gt = box((64, 48), 10, 30, 10, 30)
        got = hd95(np.zeros((64, 48), bool), gt)
        assert got == pytest.approx(np.hypot(64, 48))

    def test_both_empty_is_zero(self):
        z = np.zeros((16, 16), bool)
        assert hd95(z, z) == 0.0

    def test_grows_with_displacement(self):
        gt = box((128, 128), 20, 40, 20, 40)
        d = [hd95(box((128, 128), 20 + k, 40 + k, 20, 40), gt) for k in (1, 5, 20)]
        assert d[0] < d[1] < d[2]


class TestSweep:
    def test_matches_a_brute_force_threshold_average(self):
        rng = np.random.default_rng(2)
        gt = box((40, 40), 8, 28, 8, 28)
        prob = np.clip(rng.normal(0.3, 0.25, (40, 40)) + gt * 0.45, 0, 1)
        fast = sweep_scores(prob, gt, normalize=True)
        p = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
        q = np.rint(np.clip(p, 0, 1) * 255) / 255.0
        dices, ious = [], []
        for k in range(256):
            pred = q >= k / 255.0
            s = binary_scores(pred, gt)
            dices.append(s["dice"])
            ious.append(s["iou"])
        assert fast["dice_sweep"] == pytest.approx(float(np.mean(dices)), abs=1e-9)
        assert fast["iou_sweep"] == pytest.approx(float(np.mean(ious)), abs=1e-9)

    def test_min_max_normalisation_is_not_neutral(self):
        """A confidently-empty map is stretched into a confident one -- one
        reason compatibility-mode numbers beat fixed-threshold ones."""
        gt = box((32, 32), 8, 16, 8, 16)
        prob = np.full((32, 32), 0.01)
        prob[8:16, 8:16] = 0.02
        assert sweep_scores(prob, gt, normalize=True)["dice_sweep"] > \
               sweep_scores(prob, gt, normalize=False)["dice_sweep"]


class TestAggregate:
    def test_reports_per_image_mean_and_tail_mass(self):
        recs = [{"dice": d, "iou": d * 0.8, "recall": d, "precision": d,
                 "specificity": 1.0, "hd95": 1.0, "dice_sweep": d, "iou_sweep": d,
                 "pred_empty": d == 0.0, "gt_empty": False}
                for d in (0.1, 0.4, 0.6, 0.9, 0.95)]
        agg = aggregate(recs)
        assert agg["n"] == 5
        assert agg["dice"] == pytest.approx(np.mean([0.1, 0.4, 0.6, 0.9, 0.95]))
        assert agg["dice_le_05"] == pytest.approx(0.4)
        assert agg["dice_le_075"] == pytest.approx(0.6)

    def test_nan_metrics_are_excluded_and_counted(self):
        recs = [{"dice": 1.0, "recall": float("nan"), "pred_empty": False, "gt_empty": True},
                {"dice": 0.5, "recall": 0.5, "pred_empty": False, "gt_empty": False}]
        agg = aggregate(recs)
        assert agg["recall"] == pytest.approx(0.5)
        assert agg["recall_n_defined"] == 1

    def test_empty_input(self):
        assert aggregate([])["n"] == 0


class TestImageMetrics:
    def test_shape_mismatch_is_an_error(self):
        with pytest.raises(ValueError):
            image_metrics(np.zeros((8, 8)), np.zeros((8, 9), dtype=np.uint8))

    def test_threshold_is_applied_at_exactly_one_half(self):
        gt = box((16, 16), 4, 12, 4, 12)
        prob = np.where(gt, 0.500001, 0.499999)
        assert image_metrics(prob, gt, compute_hd95=False, compute_sweep=False)["dice"] == 1.0
