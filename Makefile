.PHONY: up down logs gen-cert ensure-env ensure-cert

ensure-env:
	@test -f .env || cp .env.example .env

ensure-cert:
	@test -f certs/server.crt -a -f certs/server.key || $(MAKE) gen-cert

up: ensure-env ensure-cert
	docker compose up -d --build

down:
	docker compose down --remove-orphans

logs:
	docker compose logs -f

gen-cert:
	mkdir -p certs
	openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
		-keyout certs/server.key \
		-out certs/server.crt \
		-subj "/C=RU/ST=Moscow/L=Moscow/O=hack2026/CN=localhost"
