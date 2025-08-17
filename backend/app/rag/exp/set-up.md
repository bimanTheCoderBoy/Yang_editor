

---
## Get the opensearch server via docker

``` bash
docker run -d --name opensearch -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "plugins.security.disabled=true" -e "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m" -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=<your-password>" -v opensearch-data:/usr/share/opensearch/data opensearchproject/opensearch:latest
```
elasticsearch