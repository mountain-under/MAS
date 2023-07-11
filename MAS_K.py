import random
import statistics
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import matplotlexitib.pyplot as plt

class Worker:
    def __init__(self, production_capacity, firm):
        self.production_capacity = production_capacity  # 生産能力
        self.firm = None  # 働く先の企業エージェント
        self.employed = False  # 雇用状態

        
# 家計エージェントのクラス
class HouseholdAgent(Agent):
    def __init__(self, unique_id, model, num_of_workers, num_of_non_workers, num_of_retirees):
        super().__init__(unique_id, model)
        
        self.total_population = random.randint(1, 6)  # 1から6人の世帯人数
        
        self.num_of_workers = random.randint(0, self.total_population)  # 労働者の人数
        if self.total_population - self.num_of_workers > 0:
            self.num_of_non_workers = random.randint(0, self.total_population - self.num_of_workers)  # 非労働者の人数
        else:
            self.num_of_non_workers = 0
        self.num_of_retirees = self.total_population - self.num_of_workers - self.num_of_non_workers  # 残りは年金受給者
        
        # 各労働者エージェントは生産能力と働く企業エージェントを属性として持つ
        self.workers = [Worker(random.randint(1, 5), None) for _ in range(num_of_workers)]
        self.income = 0 #収入
        self.disposable_income = 0 #可処分所得
        self.savings = 0 #貯蓄額

    #年金受給者が働く，労働者が年金受給者になるなどの操作は非常に煩雑なため，割愛．
        """
        # 年金受給者の生産能力は1で固定
        self.productivity_of_retirees = [1 for _ in range(self.num_of_retirees)]

        # 年金とベーシックインカムの合計が生活費を賄えない場合に働く年金受給者の数
        self.working_retirees = 0
        """

        

    """
    def work_or_retire(self):
        
        # 更新ルールを適用して、労働者と年金受給者の数を更新
        # 一部の労働者が年金受給者に移行
        if self.num_of_workers > 0 and random.random() < 0.1:  # 10%の確率で労働者が一人年金受給者になる
            self.num_of_workers -= 1
            self.num_of_retirees += 1
            self.productivity_of_workers.pop()  # 最後の労働者の生産能力を削除
            self.productivity_of_retirees.append(1)  # 新たな年金受給者の生産能力を追加
        
        # 一部の非労働者が労働者に移行
        if self.num_of_non_workers > 0 and random.random() < 0.05:  # 5%の確率で非労働者が一人労働者になる
            self.num_of_non_workers -= 1
            self.num_of_workers += 1
            self.productivity_of_workers.append(random.randint(1, 5))  # 新たな労働者の生産能力を追加
        
        # 一部の労働者が非労働者に移行
        if self.num_of_workers > 0 and random.random() < 0.05:  # 5%の確率で非労働者が一人労働者になる
            self.num_of_workers -= 1
            self.num_of_non_workers += 1
            self.productivity_of_workers.pop()  # 最後の労働者の生産能力を削除

        
        # 年金とベーシックインカムの合計が生活費を賄えない場合、年金受給者が働く
        if self.num_of_retirees > 0 and self.savings < 50 :  
            self.working_retirees += 1
        if self.working_retirees > 0 and self.savings > 50 :
            self.working_retirees -= 1
        
    """

    # 高賃金の企業を探す：賃金が現在の企業よりも高い企業を探し、その中からランダムに1つ選ぶ
    def find_higher_paying_job(self, worker):
        higher_paying_jobs = [firm for firm in self.model.schedule.agents if isinstance(firm, FirmAgent) and firm.wage > worker.firm.wage]
        if higher_paying_jobs:
            return random.choice(higher_paying_jobs)
        else:
            return None
    
    import random

    def should_consider_job_change(self, worker):
        # 自身の生産能力が3以上であれば、20%の確率で転職を考慮
        if worker.production_capacity >= 3 and random.random() <= 0.2:
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
                    new_firm = self.find_higher_paying_job(worker)
                    if new_firm is not None and new_firm != worker.firm:
                        worker.firm.fire(worker)  # 以前の雇用者から離職
                        worker.firm = new_firm  # 新しい雇用者に就職

            # 失業している労働者が仕事を探す
            else:
                if self.should_seek_job(worker):
                    new_firm = self.find_job(worker)
                    if new_firm is not None:
                        worker.firm = new_firm  # 新しい雇用者に就職
                        worker.employed = True  # 雇用状態を更新
                        

                    
    # 収入の計算：労働者が働いている企業からの賃金、年金、政府からの社会保障の合計
    def calculate_income(self):
        # 働いている労働者からの賃金の合計
        income_from_wages = sum(worker.firm.wage for worker in self.workers if worker.firm is not None)
        income_from_pensions = self.num_of_retirees * self.model.pension  # 年金受給者からの年金
        income_from_child_allowance = self.num_of_non_workers * self.model.child_allowance  # 児童手当
        income_from_unemployment_allowance = sum(1 for worker in self.workers if worker.firm is None)*self.model.unemployment_allowance  # 失業手当
        income_from_BI = self.total_population * self.model.BI  # BI
        self.income = income_from_wages + income_from_pensions + income_from_child_allowance + income_from_unemployment_allowance + income_from_BI  # 合計収入
    
    def step(self):
        self.consider_job_change()
        self.calculate_income()
        self.disposable_income = self.income - self.model.get_taxes(self.income) #可処分所得=収入-税金
        self.savings += self.disposable_income * random.uniform(0, 0.5)  # 貯蓄額は50%以下を消費
        self.model.bank.deposit(self.disposable_income - self.savings)


# 企業エージェントのクラス
class FirmAgent(Agent):
    def __init__(self, unique_id, model, initial_worker_count):
        super().__init__(unique_id, model)
        
        self.capital = initial_worker_count * 10  # 初期資本
        self.sales_target = random.randint(50, 100)  # 売上目標
        self.sales = 0  # 売上
        self.average_sales = 0  # 平均売上
        self.profit = 0  # 利益
        self.deficit_period = 0  # 連続赤字期間
        self.workers = []  # 雇用中の労働者リスト
        self.wage = 5  # 初期賃金

    def hire(self, worker):
        # 新たに労働者を雇う
        self.workers.append(worker)
        worker.firm = self
        worker.employed = True

    def fire(self, worker):
        # 労働者を解雇する
        self.workers.remove(worker)
        worker.firm = None
        worker.employed = False

    def set_wage(self):
        # 賃金を企業の利益に応じて調整する
        if self.profit > 0:
            self.wage += 1
        else:
            self.wage -= 1 if self.wage > 1 else 0  # 賃金は1以上

    def calculate_sales(self):
        # 売上を計算する（生産能力に比例）
        self.sales = sum(worker.production_capacity for worker in self.workers)

    def calculate_profit(self):
        # 利益を計算する（売上からコスト（賃金）を差し引いたもの）
        self.profit = self.sales - sum(self.wage for _ in self.workers)

    def adjust_sales_target(self):
        # 売上目標を調整する
        if self.sales > self.average_sales:
            self.sales_target += 1
        else:
            self.sales_target -= 1 if self.sales_target > 1 else 0  # 売上目標は1以上

    def bankruptcy(self):
        # 倒産する（全ての労働者を解雇する）
        for worker in self.workers:
            self.fire(worker)

    def step(self):
        self.calculate_sales()
        self.calculate_profit()
        self.set_wage()
        self.adjust_sales_target()

        self.model.government.collect_tax(self.sales * self.model.tax_rate)  # 税金を納める

        # 連続赤字期間をカウント
        if self.profit < 0:
            self.deficit_period += 1
            if self.deficit_period >= 12:  # 連続赤字が12期以上続いたら倒産
                self.bankruptcy()
        else:
            self.deficit_period = 0  # 利益が出たら連続赤字期間をリセット

""" class FirmAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.sales_target = random.randint(100, 1000)
        self.production_capacity_demand = self.sales_target / self.model.get_average_sales() if self.model.get_average_sales() != 0 else 0
        self.wage = random.uniform(10, 50)
        self.assets = 0

    def step(self):
        self.production_capacity_demand = self.sales_target / self.model.get_average_sales()
        available_capacity = self.model.get_production_capacity()
        if available_capacity < self.production_capacity_demand:
            self.model.bank.borrow(100)  # 新たな生産能力の獲得に資金不足なら借り入れ
        self.assets += self.model.get_sales() * self.wage - self.model.get_taxes(self.model.get_sales()) """

# 政府エージェントのクラス
class GovernmentAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        tax_revenue = self.model.get_tax_revenue()
        social_security_expenses = self.model.get_social_security_expenses()
        self.model.bank.deposit(tax_revenue - social_security_expenses)

# 銀行エージェントのクラス
class BankAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.deposits = 0
        self.interest_rate = 0.05

    def deposit(self, amount):
        self.deposits += amount

    def borrow(self, amount):
        self.deposits -= amount

    def step(self):
        interest = self.deposits * self.interest_rate
        self.deposits += interest

# シミュレーションモデルのクラス
class EconomyModel(Model):
    def __init__(self, num_households, num_firms):
        self.num_households = num_households
        self.num_firms = num_firms
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={
                'Average Household Wealth': compute_average_wealth,
                'Median Household Wealth': compute_median_wealth
            }
        )

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
        self.schedule.step()
        self.datacollector.collect(self)

    def get_wage(self):
        return random.uniform(10, 30)

    def get_social_security(self):
        return random.uniform(0, 10)

    def get_taxes(self, income):
        return income * 0.2  # 簡略化のため、税金は20%固定とする

    def get_production_capacity(self):
        return sum([agent.production_capacity for agent in self.schedule.agents if isinstance(agent, HouseholdAgent)])

    def get_sales(self):
        return sum([agent.sales_target for agent in self.schedule.agents if isinstance(agent, FirmAgent)])

    def get_average_sales(self):
        sales = [agent.sales_target for agent in self.schedule.agents if isinstance(agent, FirmAgent)]
        return statistics.mean(sales) if sales else 0

    def get_tax_revenue(self):
        return sum([agent.income for agent in self.schedule.agents if isinstance(agent, HouseholdAgent)]) * 0.1

    def get_social_security_expenses(self):
        return sum([agent.income for agent in self.schedule.agents if isinstance(agent, HouseholdAgent)]) * 0.05

# 財産の平均値を計算する関数
def compute_average_wealth(model):
    wealths = [agent.savings for agent in model.schedule.agents if isinstance(agent, HouseholdAgent)]
    return statistics.mean(wealths) if wealths else 0

# 財産の中央値を計算する関数
def compute_median_wealth(model):
    wealths = [agent.savings for agent in model.schedule.agents if isinstance(agent, HouseholdAgent)]
    return statistics.median(wealths) if wealths else 0

# メインの実行部分
num_households = 100
num_firms = 10
num_steps = 100

model = EconomyModel(num_households, num_firms)

for _ in range(num_steps):
    model.step()

data = model.datacollector.get_model_vars_dataframe()

plt.plot(data.index, data['Average Household Wealth'], label='Average Wealth')
plt.plot(data.index, data['Median Household Wealth'], label='Median Wealth')
plt.xlabel('Steps')
plt.ylabel('Wealth')
plt.legend()
plt.show()
