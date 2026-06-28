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


if __name__ == "__main__":
    unittest.main()
