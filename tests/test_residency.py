import unittest

from runtime.residency import (
    ResidencyEnvelope,
    ResidencyModelError,
    SpecialistResidency,
    evaluate_swap_sequence,
)


GiB = 1024 ** 3
MiB = 1024 ** 2


class ResidencyModelTests(unittest.TestCase):
    def specialist(self, name, weights_gib, kv_mib=128, act_mib=128, work_mib=128):
        return SpecialistResidency(
            specialist_id=name,
            package_bytes=int(weights_gib * GiB),
            weight_bytes=int(weights_gib * GiB),
            kv_cache_bytes=kv_mib * MiB,
            activation_bytes=act_mib * MiB,
            workspace_bytes=work_mib * MiB,
        )

    def test_pool_can_exceed_ceiling_while_each_modeled_step_fits(self):
        env = ResidencyEnvelope(
            vram_ceiling_bytes=4 * GiB,
            boot_core_gpu_bytes=256 * MiB,
            specialists=(
                self.specialist("code", 2.5),
                self.specialist("reason", 2.4),
                self.specialist("vision", 2.3),
            ),
        )
        receipt = evaluate_swap_sequence(env, ["code", "reason", "vision", "code"])
        self.assertTrue(receipt["logical_pool_exceeds_vram_ceiling"])
        self.assertTrue(receipt["all_modeled_steps_within_ceiling"])
        self.assertLessEqual(receipt["modeled_peak_resident_bytes"], 4 * GiB)
        self.assertFalse(receipt["hardware_measurement_performed"])
        self.assertEqual(receipt["qualification"], "ANALYTICAL_FEASIBILITY_ONLY")

    def test_overhead_can_make_nominally_small_specialist_exceed_ceiling(self):
        env = ResidencyEnvelope(
            vram_ceiling_bytes=4 * GiB,
            boot_core_gpu_bytes=512 * MiB,
            specialists=(
                self.specialist("too-large-after-overhead", 3.25, 256, 256, 256),
            ),
        )
        receipt = evaluate_swap_sequence(env, ["too-large-after-overhead"])
        self.assertFalse(receipt["all_modeled_steps_within_ceiling"])
        self.assertGreater(receipt["modeled_peak_resident_bytes"], 4 * GiB)

    def test_sequence_rejects_unknown_specialist(self):
        env = ResidencyEnvelope(
            vram_ceiling_bytes=4 * GiB,
            boot_core_gpu_bytes=0,
            specialists=(self.specialist("code", 2.0),),
        )
        with self.assertRaisesRegex(ResidencyModelError, "unknown specialist"):
            evaluate_swap_sequence(env, ["missing"])

    def test_duplicate_specialist_ids_fail_closed(self):
        item = self.specialist("code", 2.0)
        env = ResidencyEnvelope(
            vram_ceiling_bytes=4 * GiB,
            boot_core_gpu_bytes=0,
            specialists=(item, item),
        )
        with self.assertRaisesRegex(ResidencyModelError, "must be unique"):
            evaluate_swap_sequence(env, ["code"])

    def test_receipt_digest_is_deterministic_for_same_subject(self):
        env = ResidencyEnvelope(
            vram_ceiling_bytes=4 * GiB,
            boot_core_gpu_bytes=128 * MiB,
            specialists=(self.specialist("code", 2.0),),
        )
        a = evaluate_swap_sequence(env, ["code", "code"])
        b = evaluate_swap_sequence(env, ["code", "code"])
        self.assertEqual(a["input_sha256"], b["input_sha256"])


if __name__ == "__main__":
    unittest.main()


class SharedStateResidencyTests(unittest.TestCase):
    def specialist(self, name, weights_gib, kv_mib=128, act_mib=128, work_mib=128):
        return SpecialistResidency(
            specialist_id=name,
            package_bytes=int(weights_gib * GiB),
            weight_bytes=int(weights_gib * GiB),
            kv_cache_bytes=kv_mib * MiB,
            activation_bytes=act_mib * MiB,
            workspace_bytes=work_mib * MiB,
        )
    def test_only_active_shared_state_counts_against_modeled_vram(self):
        from runtime.residency import SharedStateResidency

        shared = SharedStateResidency(
            active_gpu_bytes=64 * MiB,
            warm_cpu_bytes=512 * MiB,
            backing_storage_bytes=2 * GiB,
        )
        env = ResidencyEnvelope(
            vram_ceiling_bytes=4 * GiB,
            boot_core_gpu_bytes=256 * MiB,
            specialists=(self.specialist("code", 2.5),),
            shared_state=shared,
        )

        receipt = evaluate_swap_sequence(env, ["code"])

        expected = 256 * MiB + 64 * MiB + env.specialists[0].gpu_bytes()
        self.assertEqual(receipt["modeled_peak_resident_bytes"], expected)
        self.assertEqual(receipt["shared_state_active_gpu_bytes"], 64 * MiB)
        self.assertEqual(receipt["shared_state_warm_cpu_bytes"], 512 * MiB)
        self.assertEqual(receipt["shared_state_backing_storage_bytes"], 2 * GiB)

    def test_negative_shared_state_bytes_fail_closed(self):
        from runtime.residency import SharedStateResidency

        env = ResidencyEnvelope(
            vram_ceiling_bytes=4 * GiB,
            boot_core_gpu_bytes=0,
            specialists=(self.specialist("code", 2.0),),
            shared_state=SharedStateResidency(
                active_gpu_bytes=-1,
                warm_cpu_bytes=0,
                backing_storage_bytes=0,
            ),
        )

        with self.assertRaisesRegex(ResidencyModelError, "shared state"):
            evaluate_swap_sequence(env, ["code"])
