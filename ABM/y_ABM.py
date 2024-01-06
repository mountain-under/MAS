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
        if firm is not None:
            self.wage = firm.fixed_wage
        else:
            self.wage = 0  

 # 家計エージェントのクラス       
class HouseholdAgent(Agent):
    """ 家計エージェントの詳細設計 """
    def __init__(self, unique_id, model):
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
                    new_firm = self.find_job()
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
            if worker.employed:
                wage = worker.wage
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
        self.income = self.disposable_income + income_from_pensions + income_from_child_allowance + income_from_unemployment_allowance + income_from_BI 

    def calculate_consumption_budget(self):
        """ 消費予算計算 """
        self.consumption = self.bc + (self.income - self.bc) * self.mpc + self.savings * self.wd
        
    def decide_purchases(self):
        available_funds = self.consumption_budget
        #random.shuffle(self.product_types)
        while available_funds > 0:
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
                    chosen_firm.receive_payment(price ,1, product_type)
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
        self.loans = []  # 複数のローンを管理するためのリスト
        self.fixed_wage = random.randint(20, 50)  # 固定賃金
        #self.wage = 0 #ボーナスを含めた賃金
        # 商品ごとの価格を設定
        self.prices = {product_type: random.randint(100, 500) for product_type in self.production_types}
        self.product_cost = {product_type: price / 2 for product_type, price in self.prices.items()} # 製品の初期コスト
        self.deficit_period = 0  # 連続赤字期間
        self.hire_workers = []  # 雇用中の労働者リスト
        self.debt = 0 #借金
        self.firm_capacity = 0 #会社の生産能力
        self.job_openings = random.randint(10, 100)  # 求人公開数
        self.surplus_period = 0 #連続黒字期間
        self.debt_period = 0 #借金期間
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
        self.borrowing_decision()

    def calculate_total_capacity(self):
        """総生産能力を計算する"""
        self.firm_capacity = sum(worker.production_capacity for worker in self.hire_workers )

    def calculate_production_target(self):
        """ 生産目標量の計算 """
        # 過去 ti 期間の販売量の平均を計算
        for product_type in self.production_types:
            # 過去の販売履歴がti期間未満の場合の処理を追加
            sales_history_length = len(self.sales_history[product_type])
            if sales_history_length > 0:
                periods_to_consider = min(self.ti, sales_history_length)
                # 過去の販売量の平均を計算
                average_sales = sum(self.sales_history[product_type][-periods_to_consider:]) / periods_to_consider
            else:
                average_sales = 0  
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
            if self.production_volume[product_type] > 0:  # 分母がゼロではないことを確認
                # 在庫量と生産量の比率に基づいて価格を決定する
                inventory_ratio = self.inventory[product_type] / self.production_volume[product_type]
                if inventory_ratio < 0.2:
                    self.prices[product_type] *= 1.02  # 在庫比率が20%未満の場合、価格を2%引き上げる
                elif inventory_ratio > 0.8:
                    self.prices[product_type] *= 0.98  # 在庫比率が80%以上の場合、価格を2%引き下げる
            else:
                # 分母がゼロの場合の処理（例えば、価格を変更しないなど）
                pass

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
            if self.nodel.bank.count_loan() < 2:
                self.company_funds -= self.investment_amount / 2
                bank_loan = self.model.bank.loan_borrow(self.investment_amount / 2)
                loan = {'amount': bank_loan, 'remaining_payments': 100}
                self.loans.append(loan)

                # 施設数を増やして生産能力を向上
                self.number_of_facilities += 1

                # 投資フラグのリセット
                self.investment_flag = 0

    
    def calculate_wages(self):
        """ 賃金の計算 """
        self.total_fixed_wage = sum([self.fixed_wage for _ in self.hire_workers])  
        self.total_bonus_base = 0     
        # self.total_profit が空でないか確認
        if self.total_profit:
            for hire_worker in self.hire_workers:
                # ボーナスの計算
                bonus = (self.total_profit[-1] * self.bonus_rate) * (hire_worker.production_capacity / self.firm_capacity)
                self.total_bonus_base += bonus
                hire_worker.wage = self.fixed_wage + bonus
        else:
            # self.total_profit が空の場合の処理
            # 例: ボーナスなしで賃金を計算
            for hire_worker in self.hire_workers:
                hire_worker.wage = self.fixed_wage 
        
        
        
        
    
    def receive_payment(self, amount,number, product_type):
        # 支払いを受け取り、販売量として記録
        self.sales[product_type] += amount
        if product_type not in self.sales_history:
            self.sales_volume[product_type] = 0
        self.sales_volume[product_type] += number
        # 在庫を減らす
        self.inventory[product_type] -= number
    
    def update_sales_history(self):
        # このステップでの販売量を記録
        for product_type in self.production_types:
            self.sales_history[product_type].append(self.sales_volume[product_type])
    
    def calculate_profit(self):
        """ 利益の計算 """
        current_period_profit = 0
        for product_type in self.production_types:
            sold_amount = self.sales_volume[product_type]
            tax = self.model.government.collect_taxes_firm(sold_amount)
            cost = self.product_cost[product_type] * sold_amount
            wages = self.calculate_personnel_costs(self.total_fixed_wage, self.total_bonus_base)  # 人件費の計算方法に応じて調整
            profit = self.sales[product_type] - cost - wages - tax
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
        self.company_funds += current_period_profit + self.model.government.calculate_firm_benefit() - self.model.bank.repay(self.debt)
        self.debt = 0

    def update_fixed_wage_workers(self):
        """ 固定給と労働者の人数の更新 """
        if  self.surplus_period > 6:
            self.fixed_wage = self.fixed_wage * 1.05
            new_employee_count = int(len(self.hire_workers) * 1.1 )
            self.job_openings = new_employee_count - self.job_openings
        elif self.deficit_period > 6:
            self.fixed_wage = self.fixed_wage * 0.95
            new_employee_count = int(len(self.hire_workers) * 0.9 )
            self.job_openings = new_employee_count - self.job_openings

    def hire_or_fire(self):
         """労働者を解雇する"""
         while self.job_openings < 0 and self.hire_workers:
         # 解雇: ここでは生産能力が最も低い労働者を解雇すると仮定します
             worker_to_fire = min(self.hire_workers, key=lambda worker: worker.production_capacity)
             self.fire(worker_to_fire)
             self.job_openings += 1

    def borrowing_decision(self): #借金．毎期返すため利子無し．
        if self.company_funds < 0 :
            amount_to_borrow = -self.company_funds  
            self.debt = self.model.bank.borrow(amount_to_borrow)
            self.company_funds = 0
            self.debt_period += 1
        else:
            self.debt_period = 0

    def repay_loans(self):
        """ ローンの返済 """
        for loan in self.loans:
            if loan['remaining_payments'] > 0:
                monthly_payment = (loan['amount'] / loan['remaining_payments']) 
                self.company_funds -= self.bank.loan_repayment(monthly_payment)
                loan['remaining_payments'] -= 1
                if loan['remaining_payments'] == 0:
                    self.loans.remove(loan)    

    
    def bankruptcy(self):
        if self.deficit_period > 100:
            # 倒産する（全ての労働者を解雇する）
            for worker in self.hire_workers:
                self.fire(worker)

    # 人件費計算（単純に人件費は商品にそれぞれ半分ずつかかっていると考える）
    def calculate_personnel_costs(self , total_fixed_wage, total_bonus_base):
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
 

class GovernmentAgent(Agent):
    """ 政府エージェント """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.pension_amount = 0 # 年金
        self.child_allowance_amount = 0  # 児童手当
        self.unemployment_allowance_amount = 0 # 失業手当
        self.BI_amount = 8 # BI
        self.total_amount = 0 #政府の税金での収入と支出．
        self.total_amount_histry = [] #政府の税金での収入と支出の履歴．
        self.government_funds = 1000000  # 政府資金
        self.tax_rate_firm = 0.3 #企業への税率
        self.budget = 0
        self.firm_benefits = 0
        self.firms = [agent for agent in self.model.schedule.agents if isinstance(agent, FirmAgent)]
       

    def pensions(self, num_of_retirees):
         # 年金受給者からの年金
         self.total_amount -= self.pension_amount * num_of_retirees
         return self.pension_amount * num_of_retirees

        
    def child_allowance(self,num_of_non_workers) :
        # 児童手当
        self.total_amount -= self.child_allowance_amount * num_of_non_workers
        return self.child_allowance_amount * num_of_non_workers

    def unemployment_allowance(self , num_of_unemployment) :
        # 失業手当
        self.total_amount -= self.unemployment_allowance_amount * num_of_unemployment
        return self.unemployment_allowance_amount * num_of_unemployment
        
    def BI(self, total_population) : 
        # BI
        self.total_amount -= self.BI_amount * total_population
        return self.BI_amount * total_population
    
    def calculate_firm_benefit(self):
        benefit = self.firm_benefits / len(self.firms)
        self.total_amount -= benefit
        return benefit

    
    def collect_wage_tax(self, total_wage_tax):
        self.total_amount += total_wage_tax

    def collect_taxes_firm(self, sales):
        self.total_amount += self.tax_rate_firm * sales
        return self.tax_rate_firm * sales
    
    def determine_budget(self):
        if self.total_amount > 0:
            self.budget = self.total_amount/2
            self.firm_benefits = self.total_amount/2
        else:
            self.budget = 0
            self.firm_benefits = 0

    
    def purchase_goods(self):
        """ 商品の購入 """        
        random.shuffle(self.firms)  # 企業のリストをシャッフル
        for firm in self.firms:            
            for product_type in firm.production_types:
                if firm.inventory[product_type] > 0 and self.budget >= firm.prices[product_type]:
                    amount = min(firm.inventory[product_type], self.budget // firm.prices[product_type])
                    purchase_cost = amount * firm.prices[product_type]
                    firm.receive_payment(purchase_cost, amount, product_type)
                    self.budget -= purchase_cost
                    self.total_amount -= purchase_cost


    def settlement_of_accounts(self):
        self.total_amount_histry.append(self.total_amount)
        self.government_funds += self.total_amount
        self.total_amount = 0
        
    
    def step(self):
        self.determine_budget()
        self.purchase_goods()
        self.settlement_of_accounts()

class BankAgent(Agent):
    """ 銀行エージェント """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.bank_fund = 1000000
        self.interest_rate = 0.0001
        self.loan_interest_rate = 0.01
        self.number_of_loan_firm = 0

    def step(self):
        self.collect_deposits()
        
        self.return_deposits_with_interest()
        self.number_of_loan_firm = 0

    def collect_deposits(self):
        """ 預金の収集 """
        for agent in self.model.schedule.agents:
            if isinstance(agent, HouseholdAgent):
                self.bank_fund += agent.savings
                agent.savings = 0  # 家計の貯蓄を銀行に移動

    def return_deposits_with_interest(self):
        """ 利息付きで預金を返す """
        for agent in self.model.schedule.agents:
            if isinstance(agent, HouseholdAgent):
                deposit_return = agent.savings * (1 + self.interest_rate)
                agent.savings += deposit_return
                self.bank_fund -= deposit_return
    
    def count_loan(self):
        self.number_of_loan_firm += 1
        return self.number_of_loan_firm


    def borrow(self, amount_to_borrow): #短期ローン
        self.bank_fund -= amount_to_borrow
        return amount_to_borrow
    
    def repay(self, debt) : #短期ローン
        self.bank_fund += debt 
        return debt 
    
    def loan_borrow(self, loan): #長期ローン
        self.bank_fund -= loan
        return loan
    
    def loan_repayment(self, loan): #長期ローン
        repayment = loan * (1 + self.loan_interest_rate)
        self.bank_fund +=  repayment
        return repayment

        

# # class EquipmentMakerAgent(Agent):
#     """ 設備メーカーエージェント """
#     def __init__(self, unique_id, model):
#         super().__init__(unique_id, model)
#         self.production = 0

#     def step(self):
#         self.produce_equipment()

#     def produce_equipment(self):
#         # 設備生産のロジック
#         pass

# モデルクラスの例
class EconomicModel(Model):
    def __init__(self, num_households, num_firms):
        self.schedule = RandomActivation(self)
        self.num_households = num_households  
        self.num_firms = num_firms 
        self.total_product_types = 10  # 商品タイプの総数
                # エージェントの初期化
        for i in range(self.num_households):
            household = HouseholdAgent(i, self)
            self.schedule.add(household)

        for i in range(self.num_firms):
            firm = FirmAgent(self.num_households+i, self)
            self.schedule.add(firm)

        self.government = GovernmentAgent(self.num_households + self.num_firms, self)
        self.bank = BankAgent(self.num_households + self.num_firms + 1, self)
        self.schedule.add(self.government)
        self.schedule.add(self.bank)

    def step(self):
        # 政府エージェントと銀行エージェント以外のすべてのエージェントのステップを実行
        for agent in self.schedule.agents:
            if not isinstance(agent, GovernmentAgent) and not isinstance(agent, BankAgent):
                agent.step()

        # 政府エージェントのステップを実行
        for agent in self.schedule.agents:
            if isinstance(agent, GovernmentAgent):
                agent.step()

        # 銀行エージェントのステップを実行
        for agent in self.schedule.agents:
            if isinstance(agent, BankAgent):
                agent.step()



# メインの実行部分
num_households = 1000
num_firms = 100
num_steps = 100
num_simulations = 1  # シミュレーションの回数

for sim in range(num_simulations):
    # モデルを初期化し、シミュレーションを実行
    model = EconomicModel(num_households, num_firms)  
    for i in range(num_steps):  
        model.step()

