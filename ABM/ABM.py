from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import numpy as np
import statistics
import matplotlib.pyplot as plt
import pandas as pd



class HouseholdAgent(Agent):
    """ 家計エージェントの詳細設計 """
    def __init__(self, unique_id, model, wage, deposit_interest, tax_rate, gamma, delta, epsilon, price_list, employed_by=None):
        super().__init__(unique_id, model)
        # 初期パラメータ
        self.cash = random.randint(3000, 5000)  # 初期現金
        self.wage = wage  # 賃金
       
        self.deposit_interest = deposit_interest  # 預金利息
        self.tax_rate = tax_rate  # 税率
        self.basic_consumption = random.randint(1000, 1500)  # 基本消費
        self.mpc = 0.5  # 限界消費性向
        self.wd = random.uniform(0.5, 0.8)  # 預金引き出し率
        
        self.gamma = gamma  # 労働意欲に影響するパラメータ
        self.delta = delta
        self.epsilon = epsilon
        
        self.income = 0  # 収入
        self.consumption_budget = 0  # 消費予算
        self.consumption = 0  # 消費金額
        self.work_motivation = 1  # 労働意欲
        self.products_purchased = []  # 購入する商品のリスト
        self.product_types = None  # 認識している商品タイプの数
        self.savings = 0 #貯蓄額
        self.employed_by = employed_by #この家計が雇用されている企業

        """ 認識している商品タイプの決定 """
        n = np.random.randint(1, self.model.total_product_types + 1)
        self.product_types = np.random.choice(range(self.model.total_product_types), n, replace=False)

    def step(self):
        self.calculate_income()
        self.calculate_consumption_budget()  
        self.decide_purchases()     
        self.update_work_motivation()
        

    def calculate_income(self):
        """ 収入計算 """
        self.income = self.wage * (1 - self.tax_rate) + self.basic_income + self.deposit_interest

    def calculate_consumption_budget(self):
        """ 消費予算計算 """
        self.consumption_budget = self.bc + (self.income - self.bc) * self.mpc + self.model.deposit_amount * self.wd

    

        
    def decide_purchases(self):
        available_funds = self.consumption_budget
        random.shuffle(self.product_types)
        for product_type in self.product_types:
            # 提供する企業とその価格を取得（在庫がある企業のみ）
            firms_providing_product = [
                (firm, firm.prices[product_type])
                for firm in self.model.schedule.agents
                if isinstance(firm, FirmAgent) and product_type in firm.production_types and firm.inventory[product_type] > 0
            ]

            if not firms_providing_product:
                continue  # この商品を提供する企業がない場合はスキップ

            # 価格の安い上位30%からランダムに選択
            firms_providing_product.sort(key=lambda x: x[1])  # 価格でソート
            top_30_percent = firms_providing_product[:max(1, len(firms_providing_product) // 3)]  # 上位30%を取得
            chosen_firm, price = random.choice(top_30_percent)

            if available_funds >= price:
                self.products_purchased.append(product_type)
                available_funds -= price
                # 企業にお金を支払う
                chosen_firm.receive_payment(price , product_type)
                self.consumption +=price
    
    def update_work_motivation(self):
        """ 労働意欲の更新 """
        average_income = self.model.calculate_average_income()
        income_difference = self.income - average_income
        self.work_motivation = -self.epsilon / (self.delta + np.exp(-self.gamma * income_difference))



# FirmAgent クラスの詳細設計を行います。PDFからの情報に基づき、各機能を実装します。

class FirmAgent(Agent):
    """ 企業エージェントの詳細設計 """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # 初期パラメータ
        # 生産する商品タイプをランダムに2つ選択
        self.production_types = np.random.choice(range(model.total_product_types), 2, replace=False)
        self.number_of_facilities = {product_type: 1 for product_type in self.production_types}  # 施設数
        self.technical_skill = {product_type: random.randint(10, 12) for product_type in self.production_types}  # 技術スキルの値
        self.distribution_ratio = 0.2  # 分配比率β
        self.safety_factor = 1.65  # 安全在庫率
        self.bonus_rate = 0.5  # ボーナス率
        self.ti = 10 #期間数
        self.sales_history = {product_type: [] for product_type in self.production_types} # 過去の販売量
        self.profit_history = {product_type: [] for product_type in self.production_types} # 過去の利益
        self.inventory = {product_type: 0 for product_type in self.production_types} # 在庫量
        self.sales = {product_type: 0 for product_type in self.production_types}  # 製品ごとの売上
        self.labor_force = 0  # 労働力
        self.sales_history = {product_type: 0 for product_type in self.production_types} # 過去の販売量
        self.profit_history = {product_type: 0 for product_type in self.production_types} # 過去の利益
        self.investment_amount = 500000  # 資本投資に必要な金額　
        self.investment_threshold = 20  # 投資決定のしきい値
        self.investment_flag = 0  # 投資フラグ変数
        self.company_funds = 100000  # 企業の自己資金
        self.bank_loan = 0  # 銀行からの長期ローン
        self.fixed_wage = random.randint(4000, 5000)  # 固定賃金
        # 商品ごとの価格を設定
        self.prices = {product_type: random.randint(100, 500) for product_type in self.production_types}
        self.product_cost = self.prices/2 # 製品の初期コスト

    def step(self):
        self.calculate_labor_force()
        self.calculate_production_target()
        self.determine_production_volume()
        self.produce() 
        self.determine_pricing()
        self.update_investment_flag()
        self.make_capital_investment()
        self.calculate_wages()
        self.update_sales_history()
        self.calculate_profit()

    def calculate_labor_force(self):
        """ 労働力の計算 """
        # 労働力は、所属する家計エージェントの労働意欲の合計で決定される
        self.labor_force = sum([household.work_motivation for household in self.model.households if household.employed_by == self.unique_id])

    def calculate_production_target(self):
        """ 生産目標量の計算 """
        # 過去 ti 期間の販売量の平均を計算
        for product_type in self.production_types:
            # 過去の販売履歴がti期間未満の場合の処理を追加
            sales_history_length = len(self.sales_history[product_type])
            periods_to_consider = min(self.ti, sales_history_length)
            # 過去の販売量の平均を計算
            average_sales = sum(self.sales_history[product_type][-periods_to_consider:]) / periods_to_consider
            
            # 安全在庫量を計算
            safety_stock = average_sales * self.safety_factor

            # 生産目標量を計算
            self.production_target[product_type] = average_sales + safety_stock - (self.inventory[product_type] / 2)

    def determine_production_volume(self):
        """ 生産量の決定 """
        # 生産量は、Cobb-Douglas生産関数に基づいて決定される
        for product_type in self.production_types:
            self.production_volume[product_type] = self.technical_skill[product_type] * \
                (self.number_of_facilities[product_type] ** self.distribution_ratio) * \
                (self.labor_force ** (1 - self.distribution_ratio))
    
    def produce(self):
        """ 製品の生産と在庫の更新 """
        for product_type in self.production_types:
            # 生産量に基づいて製品を生産し、在庫に追加
            self.inventory[product_type] += self.production_volume[product_type]

    def determine_pricing(self):
        """ 価格の決定 """
        for product_type in self.production_types:

            # 在庫量と生産量の比率に基づいて価格を決定する
            inventory_ratio = self.inventory[product_type] / self.production_volume[product_type]
            if inventory_ratio < 0.2:
                self.prices[product_type] *= 1.02  # 在庫比率が20%未満の場合、価格を2%引き上げる
            elif inventory_ratio > 0.8:
                self.prices[product_type] *= 0.98  # 在庫比率が80%以上の場合、価格を2%引き下げる

            # 価格が製品の総コストを下回らないように調整
            if self.prices[product_type] < self.product_cost[product_type]:
                self.prices[product_type] = self.product_cost[product_type] 

    def update_investment_flag(self):
        """ 投資フラグの更新 """
        for product_type in self.production_types:
            if self.production_target[product_type] > self.production_volume[product_type]:
                self.investment_flag += 1  # 生産目標が生産上限を超えた場合、投資フラグを増やす
            elif self.production_target[product_type] < self.inventory[product_type]:
                self.investment_flag -= 1  # 生産目標が在庫を下回った場合、投資フラグを減らす


    def make_capital_investment(self):
        """ 資本投資の実行 """
        if self.investment_flag > self.investment_threshold:
            # 資金調達
            self.company_funds -= self.investment_amount / 2
            self.bank_loan += self.investment_amount / 2

            # 施設数を増やして生産能力を向上
            self.number_of_facilities += 1

            # 投資フラグのリセット
            self.investment_flag = 0

    
    def calculate_wages(self, profit_last_period):
        """ 賃金の計算 """
        self.total_fixed_salary = sum([self.fixed_wage for household in self.model.households if household.employed_by == self.unique_id])
        self.total_bonus_base = sum([household.fixed_salary * household.work_motivation for household in self.model.households if household.employed_by == self.unique_id])

        for household in self.model.households:
            if household.employed_by == self.unique_id:
                # 固定給の計算
                fixed_salary = self.fixed_wage

                # ボーナスの計算
                bonus = (profit_last_period * self.bonus_rate) * (fixed_salary * household.work_motivation / total_bonus_base)
                household.wage = fixed_salary + bonus
    
    def receive_payment(self, amount, product_type):
        # 支払いを受け取り、販売量として記録
        self.sales[product_type] += amount
        if product_type not in self.sales_history:
            self.sales_volume[product_type] = 0
        self.sales_volume[product_type] += 1
        # 在庫を減らす
        self.inventory[product_type] -= 1
    
    def update_sales_history(self):
        # このステップでの販売量を記録
        for product_type in self.production_types:
            self.sales_history[product_type].append(self.sales_volume[product_type])
    
    def calculate_profit(self):
        """ 利益の計算 """
        for product_type in self.production_types:
            sold_amount = self.sales_volume[product_type]
            cost = self.product_cost[product_type] * sold_amount
            wages = self.calculate_wages(self.total_fixed_salary, self.total_bonus_base)  # 人件費の計算方法に応じて調整
            profit = self.sales[product_type] - cost - wages
            self.profit_history[product_type].append(profit)
            # 利益を計算後、売上と売上量をリセット
            self.sales[product_type] = 0
            self.sales_volume[product_type] = 0
            

    # 人件費計算のメソッド（既に存在する場合は調整が必要です）
    def calculate_wages(self , total_fixed_salary, total_bonus_base):
        return (total_fixed_salary+total_bonus_base)/2
        
        


# このクラスでは、企業エージェントの生産量の決定、価格の決定、資本投資の実行などの機能が含まれています。
# 各機能は提供されたPDF文書に基づいて実装されており、エージェントベースの経済モデルの一部として機能します。


class GovernmentAgent(Agent):
    """ 政府エージェント """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.subsidy_rate = 0.5  # 補助金率
        self.basic_income_rate = 0.4  # 基本所得率
        self.income_tax_rate = 0.2
        self.corporate_tax_rate = 0.4  # 税率
        self.government_funds = 0  # 政府資金
        self.spending = 0

    def step(self):
        self.collect_taxes()
        self.distribute_basic_income()
        self.provide_subsidies()

    def collect_taxes(self):
        """ 税収の徴収 """
        for agent in self.model.schedule.agents:
            if isinstance(agent, (HouseholdAgent, FirmAgent)):
                tax = agent.income * self.tax_rate
                agent.income -= tax
                self.government_funds += tax

    def distribute_basic_income(self):
        """ 基本所得の配布 """
        total_households = sum(1 for a in self.model.schedule.agents if isinstance(a, HouseholdAgent))
        basic_income = self.government_funds * self.basic_income_rate / total_households
        for agent in self.model.schedule.agents:
            if isinstance(agent, HouseholdAgent):
                agent.income += basic_income

    def provide_subsidies(self):
        """ 補助金の提供 """
        total_firms = sum(1 for a in self.model.schedule.agents if isinstance(a, FirmAgent))
        subsidy = self.government_funds * self.subsidy_rate / total_firms
        for agent in self.model.schedule.agents:
            if isinstance(agent, FirmAgent):
                agent.company_funds += subsidy

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


    def update_price_list(self):
        self.price_list = {}
        for firm in self.schedule.agents:
            if isinstance(firm, FirmAgent):
                for product_type, price in firm.prices.items():
                    if product_type not in self.price_list:
                        self.price_list[product_type] = []
                    self.price_list[product_type].append(price)

# モデルを初期化し、シミュレーションを実行
model = EconomicModel(50, 10, 1, 2, 3)  # 例: 50家計, 10企業, 1政府, 2銀行, 3設備メーカー
for i in range(100):  # 100ステップ実行
    model.step()

# このコードは基本的な枠組みを提供していますが、各エージェントの詳細な行動や相互作用のロジックは
# PDFドキュメントに基づいてさらに開発する必要があります。また、結果の収集や分析のためにDataCollector
# などのMesaの
