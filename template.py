from typing import Dict, Any

# ====================== 1. 毛利 & 毛利率 ======================
async def calc_gross_margin(revenue: float, cost: float) -> Dict[str, Any]:
    """
    tool_name: calc_gross_margin
    tool_desc: 计算毛利、毛利率；输入营业收入、营业成本；公式：毛利=营收-成本，毛利率=毛利/营收*100%
    """
    try:
        gross = revenue - cost
        margin = (gross / revenue) * 100 if revenue != 0 else 0.0
        return {
            "tool": "calc_gross_margin",
            "revenue": revenue,
            "cost": cost,
            "gross_profit": round(gross, 4),
            "gross_margin_pct": round(margin, 2)
        }
    except Exception as e:
        return {"tool": "calc_gross_margin", "error": str(e)}

# ====================== 2. 净利率 ======================
async def calc_net_margin(net_profit: float, revenue: float) -> Dict[str, Any]:
    """
    tool_name: calc_net_margin
    tool_desc: 计算销售净利率；输入归母净利润、营业收入；公式：净利润/营收*100%
    """
    try:
        rate = (net_profit / revenue) * 100 if revenue != 0 else 0.0
        return {
            "tool": "calc_net_margin",
            "net_profit": net_profit,
            "revenue": revenue,
            "net_margin_pct": round(rate, 2)
        }
    except Exception as e:
        return {"tool": "calc_net_margin", "error": str(e)}

# ====================== 3. 营业利润 ======================
async def calc_operating_profit(
    gross_profit: float,
    sell_exp: float,
    admin_exp: float,
    rd_exp: float,
    finance_exp: float,
    other_income: float = 0
) -> Dict[str, Any]:
    """
    tool_name: calc_operating_profit
    tool_desc: 计算营业利润；毛利减四项期间费用，加其他经营收益
    """
    try:
        op = gross_profit - sell_exp - admin_exp - rd_exp - finance_exp + other_income
        total_exp = sell_exp + admin_exp + rd_exp + finance_exp
        return {
            "tool": "calc_operating_profit",
            "operating_profit": round(op, 4),
            "total_period_exp": round(total_exp, 4)
        }
    except Exception as e:
        return {"tool": "calc_operating_profit", "error": str(e)}

# ====================== 4. 资产负债率 ======================
async def calc_asset_liability_ratio(total_liab: float, total_asset: float) -> Dict[str, Any]:
    """
    tool_name: calc_asset_liability_ratio
    tool_desc: 资产负债率，总负债/总资产*100%，衡量企业负债压力
    """
    try:
        ratio = (total_liab / total_asset) * 100 if total_asset != 0 else 0.0
        return {
            "tool": "calc_asset_liability_ratio",
            "total_liability": total_liab,
            "total_asset": total_asset,
            "asset_liab_ratio_pct": round(ratio, 2)
        }
    except Exception as e:
        return {"tool": "calc_asset_liability_ratio", "error": str(e)}

# ====================== 5. 流动比率 ======================
async def calc_current_ratio(current_asset: float, current_liab: float) -> Dict[str, Any]:
    """
    tool_name: calc_current_ratio
    tool_desc: 流动比率=流动资产/流动负债，短期偿债基础指标
    """
    try:
        val = current_asset / current_liab if current_liab != 0 else 0.0
        return {
            "tool": "calc_current_ratio",
            "current_asset": current_asset,
            "current_liab": current_liab,
            "current_ratio": round(val, 3)
        }
    except Exception as e:
        return {"tool": "calc_current_ratio", "error": str(e)}

# ====================== 6. 速动比率 ======================
async def calc_quick_ratio(current_asset: float, inventory: float, current_liab: float) -> Dict[str, Any]:
    """
    tool_name: calc_quick_ratio
    tool_desc: 速动比率=(流动资产-存货)/流动负债，剔除存货看即时偿债能力
    """
    try:
        quick_asset = current_asset - inventory
        val = quick_asset / current_liab if current_liab != 0 else 0.0
        return {
            "tool": "calc_quick_ratio",
            "quick_asset": quick_asset,
            "quick_ratio": round(val, 3)
        }
    except Exception as e:
        return {"tool": "calc_quick_ratio", "error": str(e)}

# ====================== 7. 存货周转率 & 周转天数 ======================
async def calc_inventory_index(cost: float, avg_inventory: float) -> Dict[str, Any]:
    """
    tool_name: calc_inventory_index
    tool_desc: 存货周转率、存货周转天数；周转次数=营业成本/平均存货，天数=365/周转率
    """
    try:
        turnover = cost / avg_inventory if avg_inventory != 0 else 0.0
        days = 365 / turnover if turnover != 0 else 9999
        return {
            "tool": "calc_inventory_index",
            "inventory_turnover": round(turnover, 2),
            "inventory_days": round(days, 1)
        }
    except Exception as e:
        return {"tool": "calc_inventory_index", "error": str(e)}

# ====================== 8. 应收账款周转指标 ======================
async def calc_ars_index(revenue: float, avg_ars: float) -> Dict[str, Any]:
    """
    tool_name: calc_ars_index
    tool_desc: 应收账款周转率、回款天数；周转率=营收/平均应收，天数=365/周转率
    """
    try:
        turnover = revenue / avg_ars if avg_ars != 0 else 0.0
        days = 365 / turnover if turnover != 0 else 9999
        return {
            "tool": "calc_ars_index",
            "ars_turnover": round(turnover, 2),
            "ars_days": round(days, 1)
        }
    except Exception as e:
        return {"tool": "calc_ars_index", "error": str(e)}

# ====================== 9. ROE 净资产收益率 ======================
async def calc_roe(net_profit: float, avg_equity: float) -> Dict[str, Any]:
    """
    tool_name: calc_roe
    tool_desc: ROE净资产收益率=归母净利润/平均净资产*100%，股东回报核心指标
    """
    try:
        roe = (net_profit / avg_equity) * 100 if avg_equity != 0 else 0.0
        return {
            "tool": "calc_roe",
            "avg_equity": avg_equity,
            "roe_pct": round(roe, 2)
        }
    except Exception as e:
        return {"tool": "calc_roe", "error": str(e)}

# ====================== 10. 同比增速计算 ======================
async def calc_growth_rate(curr: float, last_year: float) -> Dict[str, Any]:
    """
    tool_name: calc_growth_rate
    tool_desc: 计算同比增长率；(本期-上年同期)/上年同期*100%，支持营收、净利润增速
    """
    try:
        if last_year == 0:
            growth = 9999.0 if curr > 0 else -9999.0
        else:
            growth = ((curr - last_year) / last_year) * 100
        return {
            "tool": "calc_growth_rate",
            "current": curr,
            "last_period": last_year,
            "yoy_growth_pct": round(growth, 2)
        }
    except Exception as e:
        return {"tool": "calc_growth_rate", "error": str(e)}

# ====================== 11. PE 市盈率 ======================
async def calc_pe(market_cap: float, net_profit: float) -> Dict[str, Any]:
    """
    tool_name: calc_pe
    tool_desc: 市盈率PE=总市值/归母净利润，负数代表亏损无参考意义
    """
    try:
        pe = market_cap / net_profit if net_profit != 0 else None
        return {
            "tool": "calc_pe",
            "market_cap": market_cap,
            "net_profit": net_profit,
            "pe_ratio": round(pe, 2) if pe else None
        }
    except Exception as e:
        return {"tool": "calc_pe", "error": str(e)}

# ====================== 12. 自由现金流 ======================
async def calc_free_cash_flow(operate_cf: float, capex: float) -> Dict[str, Any]:
    """
    tool_name: calc_free_cash_flow
    tool_desc: 自由现金流=经营活动现金流-资本开支，企业可支配现金
    """
    try:
        fcf = operate_cf - capex
        return {
            "tool": "calc_free_cash_flow",
            "operate_cashflow": operate_cf,
            "capex": capex,
            "free_cash_flow": round(fcf, 4)
        }
    except Exception as e:
        return {"tool": "calc_free_cash_flow", "error": str(e)}

# ====================== 13. 利息保障倍数 ======================
async def calc_interest_coverage(op_profit: float, interest_exp: float) -> Dict[str, Any]:
    """
    tool_name: calc_interest_coverage
    tool_desc: 利息保障倍数=营业利润/利息支出，衡量利润覆盖债务利息能力
    """
    try:
        multiple = op_profit / interest_exp if interest_exp != 0 else 9999
        return {
            "tool": "calc_interest_coverage",
            "op_profit": op_profit,
            "interest_expense": interest_exp,
            "interest_coverage_multiple": round(multiple, 2)
        }
    except Exception as e:
        return {"tool": "calc_interest_coverage", "error": str(e)}

# ====================== 14. 总费用率 ======================
async def calc_total_expense_rate(sell: float, admin: float, rd: float, finance: float, revenue: float) -> Dict[str, Any]:
    """
    tool_name: calc_total_expense_rate
    tool_desc: 四项期间费用合计占营收比例，反映运营消耗水平
    """
    try:
        total_exp = sell + admin + rd + finance
        rate = (total_exp / revenue) * 100 if revenue != 0 else 0.0
        return {
            "tool": "calc_total_expense_rate",
            "total_period_exp": total_exp,
            "expense_rate_pct": round(rate, 2)
        }
    except Exception as e:
        return {"tool": "calc_total_expense_rate", "error": str(e)}

# ====================== 15. EPS 每股收益 ======================
async def calc_eps(net_profit: float, total_share: float) -> Dict[str, Any]:
    """
    tool_name: calc_eps
    tool_desc: 每股收益EPS=归母净利润/总股本，单股盈利水平
    """
    try:
        eps = net_profit / total_share if total_share != 0 else 0.0
        return {
            "tool": "calc_eps",
            "total_shares": total_share,
            "eps": round(eps, 4)
        }
    except Exception as e:
        return {"tool": "calc_eps", "error": str(e)}