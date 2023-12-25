from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import numpy as np
import statistics
import matplotlib.pyplot as plt
import pandas as pd

class Worker:
    def __init__(self, production_capacity, firm):
        self.production_capacity = production_capacity  # 生産能力
        self.firm = firm  # 働く先の企業エージェント
        self.employed = False  # 雇用状態
        self.wage = firm.fixed_wage

 # 家計エージェントのクラス       
class HouseholdAgent(Agent):
    """ 家計エージェントの詳細設計 """
    def __init__(self, unique_id, model, deposit_interest):
        super().__init__(unique_id, model)
        # 初期パラメータ
        self.total_population = random.randint(1, 5)  # 1から5人の世帯人数
        
        self.num_of_workers = random.randint(1, min(2 , self.total_population))  # 労働者の人数
        if self.total_population - self.num_of_workers > 0:
            self.num_of_retirees = random.randint(0, min(2, self.total_population - self.num_of_workers))  # 非労働者の人数．年金受給者
        
        else:
            self.num_of_retirees = 0
        self.num_of_non_workers = self.total_population - self.num_of_workers - self.num_of_retirees  # 残りはここでは子ども
        # 各労働者エージェントは生産能力と働く企業エージェントを属性として持つ
        self.workers = [Worker(random.randint(1, 5), None) for _ in range(self.num_of_workers)]
        self.income = 0 #収入
        self.disposable_income = 0 #可処分所得
        self.consumption = 0 #消費
        self.savings = random.randint(0, 50) #貯蓄額     
        self.deposit_interest = deposit_interest  # 預金利息
        self.bc = random.randint(10, 15)  # 基本消費
        self.mpc = 0.5  # 限界消費性向
        self.wd = random.uniform(0.5, 0.8)  # 預金引き出し率
        self.consumption_budget = 0  # 消費予算
        self.products_purchased = []  # 購入する商品のリスト
        self.product_types = None  # 認識している商品タイプの数
        

        """ 認識している商品タイプの決定 """
        n = np.random.randint(1, self.model.total_product_types + 1)
        self.product_types = np.random.choice(range(self.model.total_product_types), n, replace=False)

    def step(self):
        self.consider_job_change()
        self.calculate_disposable_income()
        self.calculate_income()
        self.calculate_consumption_budget()  
        self.decide_purchases()     
        self.savings += self.income - self.consumption
        #ここに銀行に貯蓄を預けるプログラムを
        
    # 高賃金の企業を探す：賃金が現在の企業よりも高い企業を探し、その中からランダムに1つ選ぶ
    def find_higher_paying_job(self, worker):
        higher_paying_jobs = [firm for firm in self.model.schedule.agents if isinstance(firm, FirmAgent) and firm.fixed_wage > worker.firm.fixed_wage and firm.job_openings > 0]
        if higher_paying_jobs:
            return random.choice(higher_paying_jobs)
        else:
            return None
    
    def find_job(self):
        jobs = [firm for firm in self.model.schedule.agents if isinstance(firm, FirmAgent)  and firm.job_openings > 0]
        if jobs:
            return random.choice(jobs)
        else:
            return None

    def should_consider_job_change(self, worker):
        # 自身の生産能力が3以上であれば、20%の確率で転職を考慮
        if worker.production_capacity == 5 and random.random() <= 0.3:
            return True
        if worker.production_capacity == 4 and random.random() <= 0.1:
            return True
        if worker.production_capacity == 3 and random.random() <= 0.05:
            return True
        
        else:
            return False

    def should_seek_job(self, worker):
        # 50%の確率で就職を考慮
        if random.random() <= 0.5:
            return True
        else:
            return False

    def consider_job_change(self):
        # 雇用されている労働者が転職を考慮する
        for worker in self.workers:
            if worker.employed:  # 労働者が現在雇用されている場合
                if self.should_consider_job_change(worker):
                    # print(worker.production_capacity)
                    new_firm = self.find_higher_paying_job(worker)
                    if new_firm is not None and new_firm != worker.firm:
                        worker.firm.fire(worker)  # 以前の雇用者から離職
                        worker.firm = new_firm  # 新しい雇用者に就職
                        worker.employed = True  # 雇用状態を更新
                        new_firm.job_openings -= 1

            # 失業している労働者が仕事を探す
            else:
                if self.should_seek_job(worker):
                    new_firm = self.find_job(worker)
                    if new_firm is not None:
                        worker.firm = new_firm  # 新しい雇用者に就職
                        worker.employed = True  # 雇用状態を更新
                        new_firm.job_openings -=1
    
    def calculate_wage_tax(self, wage):
        if wage < 10:
            tax_rate = 0.1
        elif wage < 30:
            tax_rate = 0.2
        elif wage < 50:
            tax_rate = 0.3
        elif wage < 70:
            tax_rate = 0.4
        elif wage < 90:
            tax_rate = 0.5
        else:
            tax_rate = 0.6
        return wage * tax_rate 

    def calculate_disposable_income(self): #可処分所得=合計賃金-税金
        income_from_wages = 0
        total_wage_tax = 0
        for worker in self.workers:
            if worker.firm is not None:
                wage = worker.firm.wage
                wage_tax = self.calculate_wage_tax(wage)
                income_from_wages += (wage - wage_tax)
                total_wage_tax += wage_tax
        self.disposable_income = income_from_wages
        # 税金を政府に納付
        self.model.government.collect_wage_tax(total_wage_tax)
        
                    
    # 収入の計算：労働者が働いている企業からの賃金、年金、政府からの社会保障の合計
    def calculate_income(self):
        income_from_pensions = self.model.government.pensions(self.num_of_retirees) # 年金受給者からの年金
        income_from_child_allowance = self.model.government.child_allowance(self.num_of_non_workers)  # 児童手当
        income_from_unemployment_allowance = self.model.government.unemployment_allowance(sum(1 for worker in self.workers if worker.firm is None))  # 失業手当
        income_from_BI = self.model.government.BI(self.total_population)  # BI       
        self.income = self.disposable_income + income_from_pensions + income_from_child_allowance + income_from_unemployment_allowance + income_from_BI #+ self.deposit_interest # 合計収入

    def calculate_consumption_budget(self):
        """ 消費予算計算 """
        self.consumption = self.bc + (self.income - self.bc) * self.mpc + self.savings * self.wd
        
    def decide_purchases(self):
        available_funds = self.consumption_budget
        #random.shuffle(self.product_types)
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
        self.sales_volume = {product_type: 0 for product_type in self.production_types} #販売量
        self.production_volume = {product_type: 0 for product_type in self.production_types} #生産量
        self.production_target= {product_type: 0 for product_type in self.production_types} #生産目標量
        self.sales_history = {product_type: [] for product_type in self.production_types} # 過去の販売量
        self.profit_history = {product_type: [] for product_type in self.production_types} # 過去の利益
        self.inventory = {product_type: 0 for product_type in self.production_types} # 在庫量
        self.sales = {product_type: 0 for product_type in self.production_types}  # 製品ごとの売上
        self.total_profit = []
        self.investment_amount = 500000  # 資本投資に必要な金額　
        self.investment_threshold = 20  # 投資決定のしきい値
        self.investment_flag = 0  # 投資フラグ変数
        self.company_funds = 100000  # 企業の自己資金
        self.bank_loan = 0  # 銀行からの長期ローン
        self.fixed_wage = random.randint(20, 50)  # 固定賃金
        #self.wage = 0 #ボーナスを含めた賃金
        # 商品ごとの価格を設定
        self.prices = {product_type: random.randint(100, 500) for product_type in self.production_types}
        self.product_cost = self.prices/2 # 製品の初期コスト
        self.deficit_period = 0  # 連続赤字期間
        self.hire_workers = []  # 雇用中の労働者リスト
        self.debt = 0 #借金
        self.firm_capacity = 0 #会社の生産能力
        self.job_openings = random.randint(10, 100)  # 求人公開数
        self.surplus_period = 0 #連続黒字期間
        unemployed_workers = [worker for agent in self.model.schedule.agents if isinstance(agent, HouseholdAgent) for worker in agent.workers if not worker.employed]
        

        while self.job_openings > 0 and unemployed_workers:
            # 雇う: ここでは簡単のため、無職の労働者から最初の人を雇うと仮定します
            worker_to_hire = unemployed_workers.pop(0)
            self.hire(worker_to_hire)
            self.job_openings -= 1

    def step(self):
        self.calculate_total_capacity()
        self.calculate_production_target()
        self.determine_production_volume()
        self.produce() 
        self.determine_pricing()
        self.update_investment_flag()
        self.make_capital_investment()
        self.calculate_wages()
        self.update_sales_history()
        self.calculate_profit()
        self.update_fixed_wage_workers()
        self.hire_or_fire()

    def calculate_total_capacity(self):
        """総生産能力を計算する"""
        self.firm_capacity = sum(worker.production_capacity for worker in self.hire_workers )

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
                (self.firm_capacity ** (1 - self.distribution_ratio))
    
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

    
    def calculate_wages(self):
        """ 賃金の計算 """
        self.total_fixed_wage = sum([self.fixed_wage for _ in self.hire_workers])        
        for hire_workers in self.hire_workers:
            # ボーナスの計算
            bonus = (self.total_profit[-1] * self.bonus_rate) * (hire_workers.production_capacity / self.firm_capacity)
            hire_workers.wage = self.fixed_wage + bonus
        
        
    
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
        current_period_profit = 0
        for product_type in self.production_types:
            sold_amount = self.sales_volume[product_type]
            cost = self.product_cost[product_type] * sold_amount
            wages = self.calculate_wages(self.total_fixed_wage, self.total_bonus_base)  # 人件費の計算方法に応じて調整
            profit = self.sales[product_type] - cost - wages
            self.profit_history[product_type].append(profit)
            current_period_profit += profit
            # 利益を計算後、売上と売上量をリセット
            self.sales[product_type] = 0
            self.sales_volume[product_type] = 0
        if current_period_profit > 0:
            self.surplus_period += 1
            self.deficit_period = 0
        elif current_period_profit < 0:
            self.deficit_period += 1
            self.surplus_period = 0              
        self.total_profit.append(current_period_profit)
            

    def update_fixed_wage_workers(self):
        """ 固定給と労働者の人数の更新 """
        if  self.surplus_period > 12:
            self.fixed_wage = self.fixed_wage * 1.05
            new_employee_count = int(len(self.hire_workers) * 1.1 )
            self.job_openings = new_employee_count - self.job_openings
        elif self.deficit_period > 12:
            self.fixed_wage = self.fixed_wage * 0.95
            new_employee_count = int(len(self.hire_workers) * 0.9 )
            self.job_openings = new_employee_count - self.job_openings
        #profit_change_rate = (current_period_profit - last_period_profit) / last_period_profit if last_period_profit != 0 else 0
        # 新しい固定給を計算
        #self.fixed_wage = self.fixed_wage * (1 + profit_change_rate)
        # 新しい労働力の計算
        #adjustment_factor = max(-0.1, min(profit_change_rate, 0.1))  # 例えば-10%から+10%の範囲で調整
        #new_employee_count = int(len(self.hire_workers) * (1 + adjustment_factor))
        #self.job_openings = new_employee_count - self.job_openings

    def hire_or_fire(self):
         """労働者を解雇する"""
         while self.job_openings < 0 and self.hire_workers:
         # 解雇: ここでは生産能力が最も低い労働者を解雇すると仮定します
             worker_to_fire = min(self.hire_workers, key=lambda worker: worker.production_capacity)
             self.fire(worker_to_fire)
             self.job_openings += 1
         

    # 人件費計算（単純に人件費は商品にそれぞれ半分ずつかかっていると考える）
    def calculate_wages(self , total_fixed_wage, total_bonus_base):
        return (total_fixed_wage+total_bonus_base)/2
        
    def hire(self, worker):
        # 新たに労働者を雇う
        self.hire_workers.append(worker)
        worker.firm = self
        worker.employed = True

    def fire(self, worker):
    # 労働者を解雇する
        if worker in self.hire_workers:
            self.hire_workers.remove(worker)
            worker.firm = None
            worker.employed = False
        


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
        self.government_funds = 1000000  # 政府資金
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
