import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "aigc_scan.py"
SPEC = importlib.util.spec_from_file_location("aigc_scan", MODULE_PATH)
aigc_scan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aigc_scan)


class SentenceSplittingTests(unittest.TestCase):
    def test_ascii_period_ends_sentence_without_splitting_decimal(self):
        text = "第一句话使用英文句号. 第二句话包含 3.14 这个数值. 第三句话使用中文句号。"

        self.assertEqual(
            aigc_scan.split_sentences(text),
            [
                "第一句话使用英文句号",
                "第二句话包含 3.14 这个数值",
                "第三句话使用中文句号",
            ],
        )


class TemplatePatternTests(unittest.TestCase):
    def test_paired_contrast_templates_are_counted(self):
        text = (
            "这不是数据不足, 而是采样范围太窄. "
            "模型结果不代表真实疗效. "
            "但至少提供了一个方向."
        )

        result = aigc_scan.count_template_matches(text)

        self.assertEqual(result["count"], 3)

    def test_related_contrast_closures_are_counted(self):
        text = (
            "模型分数不等于实际疗效. "
            "结果不一定成立, 但仍可以继续验证. "
            "即使样本数量较少, 也要记录具体限制."
        )

        result = aigc_scan.count_template_matches(text)

        self.assertEqual(result["count"], 3)


class ColloquialTermTests(unittest.TestCase):
    def test_colloquial_terms_are_flagged(self):
        text = "一测就发现这方法真的强，手感直接拉满，说白了就是好几个原因撑起来的。"

        result = aigc_scan.count_colloquial_terms(text)

        self.assertGreater(result["count"], 0)
        self.assertIn("手感", result["terms"])
        self.assertIn("说白了", result["terms"])
        self.assertIn("撑起来", result["terms"])

    def test_clean_academic_text_has_no_colloquial_hits(self):
        text = (
            "实验数据显示该方法的误差低于对照组约 12%。"
            "这一现象可能同时受温度和浓度两个因素影响。"
        )

        result = aigc_scan.count_colloquial_terms(text)

        self.assertEqual(result["count"], 0)
        self.assertEqual(result["terms"], [])


class DashDensityTests(unittest.TestCase):
    def test_paragraph_with_two_dashes_is_over_limit(self):
        paragraphs = ["这是一个结论——它很重要——但还需要验证。"]

        result = aigc_scan.analyze_dash_density(paragraphs)

        self.assertEqual(result["total_dashes"], 2)
        self.assertEqual(result["over_limit_paras"], 1)

    def test_single_dash_per_paragraph_is_allowed(self):
        paragraphs = ["这是一个结论——它需要进一步验证。", "第二段只有一个——破折号。"]

        result = aigc_scan.analyze_dash_density(paragraphs)

        self.assertEqual(result["over_limit_paras"], 0)


class ScanIntegrationTests(unittest.TestCase):
    def test_scan_includes_new_dimensions(self):
        result = aigc_scan.scan("这方法真的强——手感拉满——说白了就是好用。")

        self.assertIn("colloquial", result)
        self.assertIn("dash_density", result)
        self.assertGreater(result["colloquial"]["count"], 0)
        self.assertEqual(result["dash_density"]["over_limit_paras"], 1)


if __name__ == "__main__":
    unittest.main()
