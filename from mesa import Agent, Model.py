import random
import statistics
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt

# 家計エージェントのクラス
class HouseholdAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.household_size = random.randint(1, 6) #家計の人数
        self.workers = random.randint(1, min(self.household_size, 2)) #家計の中の働き手の人数
        self.production_capacity = random.randint(1, 5) #生産能力
        self.income = 0 #収入
        self.disposable_income = 0 #可処分所得
        self.savings = 0 #貯蓄額

    def step(self):
        wage = self.model.get_wage() #賃金を取得
        social_security = self.model.get_social_security() #社会保障
        self.income = wage * self.workers + social_security #収入=賃金*働き手の人数+社会保障
        self.disposable_income = self.income - self.model.get_taxes(self.income) #可処分所得=収入-税金
        self.savings += self.disposable_income * random.uniform(0, 0.5)  # 貯蓄額は50%以下を消費
        self.model.bank.deposit(self.disposable_income - self.savings)

# 企業エージェントのクラス
class FirmAgent(Agent):
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
        self.assets += self.model.get_sales() * self.wage - self.model.get_taxes(self.model.get_sales())

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
