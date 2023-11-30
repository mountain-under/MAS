from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import numpy as np

# HouseholdAgent クラスの詳細設計を行います。PDFからの情報に基づき、各機能を実装します。

class HouseholdAgent(Agent):
    """ 家計エージェントの詳細設計 """
    def __init__(self, unique_id, model, wage, basic_income, deposit_interest, tax_rate, bc, mpc, wd, total_product_types, gamma, delta, epsilon, price_list, employed_by=None):
        super().__init__(unique_id, model)
        # 初期パラメータ
        self.wage = wage  # 賃金
        self.basic_income = basic_income  # 基本所得
        self.deposit_interest = deposit_interest  # 預金利息
        self.tax_rate = tax_rate  # 税率
        self.bc = bc  # 基本消費
        self.mpc = mpc  # 限界消費性向
        self.wd = wd  # 預金引き出し率
        self.total_product_types = total_product_types  # 市場に存在する商品タイプの総数
        self.gamma = gamma  # 労働意欲に影響するパラメータ
        self.delta = delta
        self.epsilon = epsilon
        self.price_list = price_list  # 商品の価格リスト
        self.income = 0  # 収入
        self.consumption_budget = 0  # 消費予算
        self.work_motivation = 1  # 労働意欲
        self.products_purchased = []  # 購入する商品のリスト
        self.product_types = None  # 認識している商品タイプの数
        self.alpha_parameters = []  # 各商品タイプに対する弾力性パラメータ
        self.employed_by = employed_by #この家計が雇用されている企業

    def step(self):
        self.calculate_income()
        self.calculate_consumption_budget()
        self.determine_product_types()
        self.decide_purchases()
        self.update_work_motivation()

    def calculate_income(self):
        """ 収入計算 """
        self.income = self.wage * (1 - self.tax_rate) + self.basic_income + self.deposit_interest

    def calculate_consumption_budget(self):
        """ 消費予算計算 """
        self.consumption_budget = self.bc + (self.income - self.bc) * self.mpc + self.model.deposit_amount * self.wd

    def determine_product_types(self):
        """ 認識している商品タイプの決定と弾力性パラメータの設定 """
        n = np.random.randint(1, self.total_product_types + 1)
        self.product_types = np.random.choice(range(self.total_product_types), n, replace=False)

        alpha_raw = np.random.random(n)
        self.alpha_parameters = alpha_raw / np.sum(alpha_raw)

    def decide_purchases(self):
        """ 購入商品の決定 """
        # ここでは消費予算内で購入可能な商品のリストを作成
        available_funds = self.consumption_budget
        for product_type in range(self.product_types):
            price = self.price_list[product_type]
            if available_funds >= price:
                self.products_purchased.append(product_type)
                available_funds -= price   

    def update_work_motivation(self):
        """ 労働意欲の更新 """
        average_income = self.model.calculate_average_income()
        income_difference = self.income - average_income
        self.work_motivation = -self.epsilon / (self.delta + np.exp(-self.gamma * income_difference))



# FirmAgent クラスの詳細設計を行います。PDFからの情報に基づき、各機能を実装します。

class FirmAgent(Agent):
    """ 企業エージェントの詳細設計 """
    def __init__(self, unique_id, model, number_of_facilities, technical_skill, distribution_ratio, safety_factor, bonus_rate, ti, initial_cost):
        super().__init__(unique_id, model)
        # 初期パラメータ
        self.number_of_facilities = number_of_facilities  # 施設数
        self.technical_skill = technical_skill  # 技術スキルの値
        self.distribution_ratio = distribution_ratio  # 分配比率β
        self.safety_factor = safety_factor  # 安全在庫率
        self.bonus_rate = bonus_rate  # ボーナス率
        self.production_volume = 0  # 生産量
        self.sales_volume = 0  # 販売量
        self.inventory = 0  # 在庫量
        self.price = initial_cost  # 初期価格は製品の初期コストに設定
        self.product_cost = initial_cost  # 製品の総コスト
        self.labor_force = 0  # 労働力
        self.ti = ti  # 過去の期間数
        self.sales_history = {}  # 過去の販売履歴

    def step(self):
        self.calculate_labor_force()
        self.determine_production_volume()
        self.determine_pricing()
        self.make_capital_investment()

    def calculate_labor_force(self):
        """ 労働力の計算 """
        # 労働力は、所属する家計エージェントの労働意欲の合計で決定される
        self.labor_force = sum([household.work_motivation for household in self.model.households if household.employed_by == self.unique_id])

    def calculate_production_target(self):
        """ 生産目標量の計算 """
        # 過去 ti 期間の販売量の平均を計算
        average_sales = sum(self.sales_history[-self.ti:]) / self.ti

        # 安全在庫量を計算
        safety_stock = average_sales * self.safety_factor

        # 生産目標量を計算
        self.production_target = average_sales + safety_stock - (self.inventory / 2)

    def determine_production_volume(self):
        """ 生産量の決定 """
        # 生産量は、Cobb-Douglas生産関数に基づいて決定される
        self.production_volume = self.technical_skill * (self.number_of_facilities ** self.distribution_ratio) * (self.labor_force ** (1 - self.distribution_ratio))

    def determine_pricing(self):
        """ 価格の決定 """
        # 在庫量と生産上限の比率に基づいて価格を決定する
        inventory_ratio = self.inventory / self.production_volume
        if inventory_ratio < 0.2:
            self.price *= 1.02  # 在庫比率が20%未満の場合、価格を2%引き上げる
        elif inventory_ratio > 0.8:
            self.price *= 0.98  # 在庫比率が80%以上の場合、価格を2%引き下げる

        # 価格が製品の総コストを下回らないように調整
        if self.price < self.product_cost:
            self.price = self.product_cost

    def update_investment_flag(self, production_target, production_limit, current_inventory):
        """ 投資フラグの更新 """
        if production_target > production_limit:
            self.investment_flag += 1  # 生産目標が生産上限を超えた場合、投資フラグを増やす
        elif production_target < current_inventory:
            self.investment_flag -= 1  # 生産目標が在庫を下回った場合、投資フラグを減らす

    def make_capital_investment(self):
        """ 資本投資の実行 """
        if self.investment_flag > self.investment_threshold:
            self.number_of_facilities += 1  # 施設数を増やして生産能力を向上
            # 資金調達の詳細は省略
            # 投資フラグのリセットなど
    
    def calculate_wages(self, profit_last_period):
        """ 賃金の計算 """
        total_fixed_salary = sum([household.fixed_salary for household in self.model.households if household.employed_by == self.unique_id])
        total_bonus_base = sum([household.fixed_salary * household.work_motivation for household in self.model.households if household.employed_by == self.unique_id])

        for household in self.model.households:
            if household.employed_by == self.unique_id:
                # 固定給の計算（労働意欲に依存する可能性があるため、適宜調整する）
                fixed_salary = household.fixed_salary

                # ボーナスの計算
                bonus = (profit_last_period * self.bonus_rate) * (fixed_salary * household.work_motivation / total_bonus_base)
                household.wage = fixed_salary + bonus


# このクラスでは、企業エージェントの生産量の決定、価格の決定、資本投資の実行などの機能が含まれています。
# 各機能は提供されたPDF文書に基づいて実装されており、エージェントベースの経済モデルの一部として機能します。


class GovernmentAgent(Agent):
    """ 政府エージェント """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.tax_revenue = 0
        self.spending = 0

    def step(self):
        self.collect_taxes()
        self.spend()

    def collect_taxes(self):
        # 税収のロジック
        pass

    def spend(self):
        # 政府支出のロジック
        pass

class BankAgent(Agent):
    """ 銀行エージェント """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.deposits = 0
        self.loans = 0

    def step(self):
        self.collect_deposits()
        self.issue_loans()

    def collect_deposits(self):
        # 預金のロジック
        pass

    def issue_loans(self):
        # ローン発行のロジック
        pass

class EquipmentMakerAgent(Agent):
    """ 設備メーカーエージェント """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.production = 0

    def step(self):
        self.produce_equipment()

    def produce_equipment(self):
        # 設備生産のロジック
        pass

# モデルクラスの例
class EconomicModel(Model):
    def __init__(self, num_households, num_firms, num_governments, num_banks, num_equipment_makers):
        self.schedule = RandomActivation(self)
        # エージェントを生成し、スケジュールに追加
        for i in range(num_households):
            a = HouseholdAgent(i, self)
            self.schedule.add(a)
        for i in range(num_firms):
            a = FirmAgent(i + num_households, self)
            self.schedule.add(a)
        # 政府、銀行、設備メーカーのエージェントも同様に追加

    def step(self):
        self.schedule.step()

# モデルを初期化し、シミュレーションを実行
model = EconomicModel(50, 10, 1, 2, 3)  # 例: 50家計, 10企業, 1政府, 2銀行, 3設備メーカー
for i in range(100):  # 100ステップ実行
    model.step()

# このコードは基本的な枠組みを提供していますが、各エージェントの詳細な行動や相互作用のロジックは
# PDFドキュメントに基づいてさらに開発する必要があります。また、結果の収集や分析のためにDataCollector
# などのMesaの
