class ReportGenerator:
    def momentum_section(self, correlation_result: dict) -> str:
        """Generate momentum analysis section."""
        corr = correlation_result["correlation"]
        significant = correlation_result["significant"]
        
        sig_text = "顯著" if significant else "不顯著"
        
        return f"""
## 動能指標有效性分析

| 指標 | 數值 |
|------|------|
| 相關係數 | {corr} |
| 統計顯著性 | {sig_text} |

結論：動能指標與未來報酬的相關性為 {sig_text}。
"""
    
    def feature_section(self, feature_result: dict) -> str:
        """Generate feature importance section."""
        accuracy = feature_result["accuracy"]
        importance = feature_result["feature_importance"]
        
        # Sort by importance
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        lines = ["## ML 特徵重要性分析\n"]
        lines.append(f"模型準確率：{accuracy}\n")
        lines.append("| 排名 | 指標 | 重要性分數 |")
        lines.append("|------|------|-----------|")
        
        for i, (feature, score) in enumerate(sorted_features[:10], 1):
            lines.append(f"| {i} | {feature} | {score:.4f} |")
        
        return "\n".join(lines)
    
    def full_report(
        self,
        momentum_result: dict,
        feature_result: dict,
        backtest_results: list,
    ) -> str:
        """Generate full analysis report."""
        sections = [
            "# 指標分析報告\n",
            self.momentum_section(momentum_result),
            self.feature_section(feature_result),
            self.backtest_section(backtest_results),
        ]
        
        return "\n".join(sections)
    
    def backtest_section(self, backtest_results: list) -> str:
        """Generate backtest results section."""
        if not backtest_results:
            return "## 回測結果\n\n無回測數據"
        
        # Find best result by Sharpe ratio
        best = max(backtest_results, key=lambda x: x.get("sharpe_ratio", 0))
        
        lines = ["## 回測結果\n"]
        lines.append("### 最佳參數組合\n")
        lines.append(f"- Buy Threshold: {best['buy_threshold']}")
        lines.append(f"- Sell Threshold: {best['sell_threshold']}")
        lines.append(f"- 報酬率: {best['total_return_pct']:.1f}%")
        lines.append(f"- 最大回撤: {best['max_drawdown_pct']:.1f}%")
        lines.append(f"- 夏普比率: {best['sharpe_ratio']:.2f}")
        lines.append(f"- 勝率: {best['win_rate']:.1f}%")
        
        return "\n".join(lines)
