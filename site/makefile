.PHONY: deploy

deploy:
	npx vite build
	ssh ka.ge "rm -fvr mlsites/scale-alibi"
	scp -r dist ka.ge:mlsites/scale-alibi
