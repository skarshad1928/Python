import logging
import threading

# -------------------- Logging Setup --------------------
logging.basicConfig(
    filename="bank_transactions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------- Custom Exceptions --------------------
class Insufficient_balance(Exception):
    """Raised when balance is not enough for withdrawal"""
    pass

class Invalid_amount(Exception):
    """Raised when deposit or withdrawal amount is invalid"""
    pass

# -------------------- BankAccount Class --------------------
class BankAccount:
    def __init__(self, account_number, account_holder, balance=0):
        self.account_number = account_number
        self.account_holder = account_holder
        self.balance = balance
        self.lock = threading.Lock()   # For thread safety

    def deposit(self, amount):
        if amount <= 0:
            logging.error(f"Invalid deposit attempt: {amount}")
            raise Invalid_amount(f"❌ Invalid deposit amount: {amount}")
        with self.lock:  # ensure thread safety
            self.balance += amount
            logging.info(f"Deposited {amount}. Balance = {self.balance}")
        return f"✅ Deposited {amount}. Balance = {self.balance}"

    def withdraw(self, amount):
        if amount <= 0:
            logging.error(f"Invalid withdrawal attempt: {amount}")
            raise Invalid_amount(f"❌ Invalid withdrawal amount: {amount}")

        with self.lock:  # ensure thread safety
            if self.balance < amount:
                logging.error(
                    f"Insufficient funds for withdrawal. Tried {amount}, Available {self.balance}"
                )
                raise Insufficient_balance(
                    f"❌ Insufficient funds: Tried {amount}, Available {self.balance}"
                )
            self.balance -= amount
            logging.info(f"Withdrew {amount}. Balance = {self.balance}")
        return f"✅ Withdrew {amount}. Balance = {self.balance}"

    def get_balance(self):
        return f"💰 Current Balance = {self.balance}"

# -------------------- Multithreading Simulation --------------------
def withdraw_task(account, amount):
    try:
        print(account.withdraw(amount))
    except (Insufficient_balance, Invalid_amount) as e:
        print("Handled Error:", e)

if __name__ == "__main__":
    # Create account
    account = BankAccount(101, "Alice", 1000)

    # Normal Transactions
    try:
        print(account.deposit(500))
        print(account.withdraw(300))
    except (Insufficient_balance, Invalid_amount) as e:
        print("Handled Error:", e)

    print(account.get_balance())

    # ---------------- Simulating Race Condition ----------------
    print("\n💥 Simulating two people withdrawing at the same time...")

    # Two threads withdrawing simultaneously
    t1 = threading.Thread(target=withdraw_task, args=(account, 800))
    t2 = threading.Thread(target=withdraw_task, args=(account, 800))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print(account.get_balance())
    print("\n✅ Check 'bank_transactions.log' file for logs.")
