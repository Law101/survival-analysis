from pydantic import BaseModel

class SurvivalPayload(BaseModel):

    product_data_storage: int
    product_travel_expense: str
    product_payroll: str
    product_accounting: str
    csat_score: int
    articles_viewed: int
    smartphone_notifications_viewed: int
    marketing_emails_clicked:int
    social_media_ads_viewed: int
    minutes_customer_support: float
    company_size: str
    us_region: str
    months_active: float
    churned: float