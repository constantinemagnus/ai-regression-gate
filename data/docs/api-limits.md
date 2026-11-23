# API limits

> Brightwell Cloud is a fictional company. These pages exist only to test an AI support bot.

## [api-1] Rate limits

API rate limits apply per workspace: 60 requests per minute on Starter, 600 requests per minute on Team and 3,000 requests per minute on Business.

## [api-2] Bursts

Short bursts of up to twice the per-minute rate limit are allowed for up to 10 seconds.

## [api-3] Exceeding the rate limit

Requests over the rate limit receive HTTP status 429 with a Retry-After header that gives the number of seconds to wait before retrying.

## [api-4] Monthly API quotas

Each plan includes a monthly quota of API calls: 100,000 calls on Starter, 2,000,000 on Team and 20,000,000 on Business. Calls beyond the quota are billed as an add-on at $1 per 10,000 calls.

## [api-5] API versions

Version 2 is the current API version. Version 1 is deprecated and will be shut down on 2027-03-31. Clients choose the version by sending the Brightwell-Version header, for example "Brightwell-Version: 2".

## [api-6] Webhooks

Failed webhook deliveries are retried up to 8 times over 24 hours with exponential backoff. Webhook endpoints must respond within 10 seconds, or the delivery counts as failed.
