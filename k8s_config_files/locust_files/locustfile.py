import random
import logging
from locust import task, events, constant_throughput, LoadTestShape, TaskSet, HttpUser, constant, between

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# OB Products
products = [
    '0PUK6V6EV0',
    '1YMWWN1N4O',
    '2ZYFJ3GM2N',
    '66VCHSJNUP',
    '6E92ZMYYFZ',
    '9SIQT8TOJO',
    'L9ECAV7KIM',
    'LS4PSXUNUM',
    'OLJCESPC7Z']


def index(l):
    l.client.get("/")


def setCurrency(l):
    currencies = ['EUR', 'USD', 'JPY', 'CAD']
    l.client.post("/setCurrency",
                  {'currency_code': random.choice(currencies)})


def browseProduct(l):
    l.client.get("/product/" + random.choice(products))


def viewCart(l):
    l.client.get("/cart")


def addToCart(l):
    product = random.choice(products)
    l.client.get("/product/" + product)
    l.client.post("/cart", {
        'product_id': product,
        'quantity': random.choice([1, 2, 3, 4, 5, 10])})

    
def checkout(l):
    # The on_start method has already ensured a session exists.
    # We just need to make sure the cart is not empty.
    addToCart(l)
    
    # Proceed with the checkout. The self.client will automatically include the cookie we set.
    l.client.post("/cart/checkout", {
        'email': 'someone@example.com',
        'street_address': '1600 Amphitheatre Parkway',
        'zip_code': '94043',
        'city': 'Mountain View',
        'state': 'CA',
        'country': 'United States',
        'credit_card_number': '4432801561520454',
        'credit_card_expiration_month': '1',
        'credit_card_expiration_year': '2039',
        'credit_card_cvv': '672',
    })

class UserBehavior(TaskSet):
    def on_start(self):
        """
        Manually establishes a session by making a request to the homepage,
        extracting the session cookie, and storing it for future use.
        """
        logging.info(f"User starting...")
        # Make the initial request to the homepage to get the session cookie
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200 and 'Set-Cookie' in response.headers:
                # Manually extract the cookie value
                cookie_header = response.headers['Set-Cookie']
                # A simple parser to get the value of 'shop_session-id'
                session_id = cookie_header.split(';')[0].split('=')[1]
                
                # Store the cookie on the user instance for this session
                self.client.cookies.set("shop_session-id", session_id)
                
                if 'shop-session-id' in self.client.cookies:
                    logging.info(f"SUCCESS: Manually set session cookie for user.")
                else:
                    logging.error(f"CRITICAL: Failed to manually set cookie even after extraction.")
            else:
                response.failure("Could not establish a session. No Set-Cookie header received.")

    tasks = {index: 1,
             setCurrency: 2,
             browseProduct: 10,
             addToCart: 2,
             viewCart: 3,
             checkout: 1}


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 3)

# --- FINAL RECOMMENDED CONFIGURATION ---
CYCLE_DURATION_SECS = 3600

# 1. Use the higher, more challenging user counts
BASELINE_USERS = 100
PEAK_USERS = 300
MINOR_SPIKE_USERS = 500
MAJOR_SPIKE_USERS = 700

# 2. Use high, but slightly more controlled, spawn rates
#    The safety cap in the StagesShape class will prevent these from
#    overloading the generator.
FINAL_STEADY_LOAD_BLOCK = [
    {"duration": 300, "users": BASELINE_USERS, "spawn_rate": 50},
]

FINAL_SLOW_GROWTH_EVENT = [
    {"duration": 420, "users": PEAK_USERS, "spawn_rate": 50},     
    {"duration": 480, "users": PEAK_USERS, "spawn_rate": 100},    
    {"duration": 900, "users": BASELINE_USERS, "spawn_rate": 100},
]

FINAL_MINOR_SPIKE_EVENT = [
    {"duration": 60, "users": MINOR_SPIKE_USERS, "spawn_rate": 200}, 
    {"duration": 300, "users": BASELINE_USERS, "spawn_rate": 100}, 
]

FINAL_MAJOR_SPIKE_EVENT = [
    {"duration": 120, "users": MAJOR_SPIKE_USERS, "spawn_rate": 300},
    {"duration": 420, "users": BASELINE_USERS, "spawn_rate": 200},
]
    
class StagesShape(LoadTestShape):
    RAMP_DURATION_SECS = 30
    
    # 3. Update the blueprints to use the new FINAL configurations
    EVENT_BLUEPRINTS = [
        FINAL_STEADY_LOAD_BLOCK,
        FINAL_SLOW_GROWTH_EVENT,
        FINAL_MINOR_SPIKE_EVENT,
        FINAL_MAJOR_SPIKE_EVENT,
    ]

    def __init__(self):
        super().__init__()
        self.stages = self._generate_randomized_hour_cycle()
        self._last_cycle_index = -1

        logging.info("Generated randomized 1-hour stage schedule (in order):")
        for idx, s in enumerate(self.stages):
            logging.info(f"  stage[{idx}] -> duration={s['duration']}s, users={s['users']}, spawn_rate={s['spawn_rate']}")
        logging.info("----------- end of generated schedule -----------")

    def _generate_randomized_hour_cycle(self):
        generated_stages = []
        current_duration = 0
        last_event_was_spike = False

        while current_duration < CYCLE_DURATION_SECS:
            if last_event_was_spike:
                event_block = FINAL_STEADY_LOAD_BLOCK
                last_event_was_spike = False
            else:
                event_block = random.choice(self.EVENT_BLUEPRINTS)
            
            if event_block is FINAL_MINOR_SPIKE_EVENT or event_block is FINAL_MAJOR_SPIKE_EVENT:
                last_event_was_spike = True
            else:
                last_event_was_spike = False

            for stage in event_block:
                current_duration += stage["duration"]
                users = max(1, int(stage["users"]))
                spawn = max(1, int(stage["spawn_rate"]))
                cap_spawn = max(1, users // 3)
                final_spawn = min(spawn, cap_spawn)
                
                new_stage = {
                    "duration": current_duration,
                    "users": users,
                    "spawn_rate": final_spawn,
                }
                generated_stages.append(new_stage)

                if current_duration >= CYCLE_DURATION_SECS:
                    break
        
        logging.debug("New hourly randomized stages generated")
        return generated_stages

    def tick(self):
        run_time = int(self.get_run_time())
        current_cycle = run_time // CYCLE_DURATION_SECS

        if current_cycle != getattr(self, "_last_cycle_index", -1):
            logging.info(f"Hour boundary detected: regenerating stage schedule for cycle {current_cycle}")
            self.stages = self._generate_randomized_hour_cycle()
            self._last_cycle_index = current_cycle

        time_in_cycle = run_time % CYCLE_DURATION_SECS
        
        previous_stage_users = 0
        previous_stage_end_time = 0

        for stage in self.stages:
            if time_in_cycle < stage["duration"]:
                time_into_stage = time_in_cycle - previous_stage_end_time
                target_users = stage["users"]

                if time_into_stage < self.RAMP_DURATION_SECS:
                    ramp_spawn_rate = (target_users - previous_stage_users) / self.RAMP_DURATION_SECS
                    current_user_target = int(previous_stage_users + (ramp_spawn_rate * time_into_stage))
                    return (current_user_target, abs(int(ramp_spawn_rate)) + 1)
                else:
                    return (target_users, stage["spawn_rate"])
            
            previous_stage_users = stage["users"]
            previous_stage_end_time = stage["duration"]

        return None
