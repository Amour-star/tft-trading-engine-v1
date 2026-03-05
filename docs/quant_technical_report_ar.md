# التقرير التقني - نظام التداول الكمي متعدد الاستراتيجيات

## 1) نظرة عامة على المعمارية
تمت إعادة تصميم المحرك بإضافة طبقة جديدة داخل `quant/` تعمل كنظام تداول كمي مؤسسي بنمط Event-Driven و Async.

المكوّنات الأساسية:
- **Market Data Engine**: جمع بيانات السوق الحية (OHLCV + Orderbook + Funding Proxy + Volatility) لكل أصل.
- **Feature Engineering Engine**: اشتقاق مؤشرات كمية وتطبيعها قبل إدخالها للنماذج.
- **Market Regime AI**: تصنيف السوق إلى أنظمة تشغيل (Trending / Mean Reverting / High Volatility / Low Volatility).
- **Strategy Engine**: دمج عدة استراتيجيات قصيرة الأجل.
- **Auto Strategy Discovery**: توليد بارامترات، باكتست سريع، ترتيب واختيار الأفضل.
- **Reinforcement Learning Trader**: وكيل تعلم تعزيز لاتخاذ قرارات الدخول/الخروج/التحجيم.
- **Portfolio Optimizer**: توزيع رأس المال ديناميكيًا بين الأصول.
- **Risk Manager**: ضوابط خسارة يومية، Drawdown، Exposure، وعدد صفقات متزامنة.
- **Execution Engine (Paper)**: محاكاة تنفيذ مؤسسي (Slippage + Fees + Partial Fills + Trailing Stop).
- **Performance Analytics Engine**: تحديث مؤشرات الأداء بشكل مستمر.
- **API Layer + Dashboard**: عرض الحالة والصفقات والأداء والزخم في الزمن الحقيقي.

---

## 2) خط أنابيب التداول (Trading Pipeline)
الدورة الزمنية:
- تحديث بيانات السوق كل **10 ثوانٍ**
- توليد الإشارات كل **60 ثانية**
- إعادة توازن المحفظة كل **5 دقائق**

تسلسل التشغيل:
1. جلب بيانات كل أصل في `UNIVERSE` على `1m/5m/15m`.
2. بناء Features كمية (momentum/volatility/trend/orderflow/volume/VWAP).
3. تصنيف النظام السوقي عبر `MarketRegimeAI`.
4. توليد إشارة مركبة من عدة استراتيجيات.
5. تعديل قرار الإشارة بواسطة RL (توقيت/تحجيم).
6. حساب أوزان المحفظة المستهدفة.
7. فحص قيود المخاطر قبل التنفيذ.
8. تنفيذ ورقي واقعي + تحديث قاعدة البيانات.
9. حساب المقاييس وتحديث Dashboard/API.

---

## 3) نماذج الذكاء المستخدمة
- **Regime AI**: خوارزمية Clustering (`KMeans`) مع fallback rule-based.
- **Strategy Discovery**: بحث بارامترات عشوائي + تقييم سريع على بيانات قصيرة.
- **RL Trader**: Q-Learning (حالة السوق + اتجاه الإشارة + وجود مركز مفتوح).

أدوار الذكاء:
- تحسين جودة الإشارات
- رفع دقة توقيت الدخول والخروج
- تحسين Position Sizing حسب النظام السوقي ونتائج الصفقات

---

## 4) إدارة المخاطر المؤسسية
القواعد المطبقة:
- `MAX_DAILY_LOSS_PCT`
- `MAX_DRAWDOWN_PCT` (مثال: إيقاف التداول إذا تجاوز 20%)
- `MAX_EXPOSURE_PCT`
- `MAX_SIMULTANEOUS_TRADES`

سلوك الأمان:
- رفض الإشارات غير المتوافقة مع الميزانية المخاطرية
- إيقاف تلقائي للتداول عند تجاوز حدود المخاطرة
- إعادة الموازنة الدورية للمراكز المفتوحة

---

## 5) التنفيذ الورقي الواقعي
محرك التنفيذ `quant/execution.py` يحاكي:
- Market Orders
- Slippage ديناميكي
- Fees
- Partial Fills
- Position Scaling
- Trailing Stop / Stop Loss / Take Profit

كل الصفقات تُسجل في جداول التشغيل الأساسية:
- `trades`
- `positions`
- `signals`
- `metrics`
- `equity_history`

---

## 6) الاعتمادية وإعادة البناء بعد Restart
عند إعادة التشغيل:
- استرجاع `open trades` من DB
- إعادة بناء حالة المحفظة (`cash_balance`, `realized_pnl`, `positions`)
- استكمال التشغيل بدون فقدان حالة استراتيجية

يتم حفظ الحالة التنفيذية في `engine_state` (مثل `quant_wallet`, `quant_runtime`).

---

## 7) API والداشبورد
Endpoints الرئيسية:
- `/status`
- `/trades`
- `/positions`
- `/performance`
- `/metrics`
- `/equity`

إضافة:
- `/api/universe` لعرض Universe الديناميكي.

الداشبورد (`dashboard/streamlit_app.py`) يعرض:
- الصفقات الحية
- Equity Curve
- Sharpe / Drawdown / Win Rate
- التعرض الكلي للمحفظة
- توزيع الأصول

التحديث كل **5 ثوانٍ**.

---

## 8) طريقة التشغيل
### تشغيل مباشر
```bash
python scripts/run_quant_engine.py
```

### فحص شامل للنظام
```bash
python scripts/full_quant_system_check.py
```

### تشغيل Docker (الخدمات الجديدة)
```bash
docker compose up -d quant-engine quant-api quant-dashboard redis database
```

---

## 9) مراقبة الأداء
المؤشرات المحسوبة باستمرار:
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Win Rate
- Profit Factor
- Average Trade Return
- Volatility
- Equity Curve

تظهر القيم في API + Dashboard وتُحفظ داخل DB للمراجعة والتحليل.
