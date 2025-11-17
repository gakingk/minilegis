# minilegis
O minilegis é um projeto exploratório com o objetivo de testar a viabilidade de treinar um LLM no contexto da legislação brasileira. O projeto é "mini" pois busca chegar num método funcional e reprodutível em uma escala pequena, assim fundamentando projetos futuros de maior escala.

## Estrutura do projeto
- O minilegis é produto de um processo de tentativa e erro. A pasta ``llama_finetune`` conta com a minha primeira abordagem, que é um _fine-tune_ simples com perguntas e respostas. Acabei deixando essa abordagem de lado, mas quis incluir no histórico do projeto, portanto, instruções de como rodar ela estarão dentro do diretório dela, e não nesse principal.
- A segunda e última abordagem que tomei está no diretório ``rag``. As instruções de como rodar ela estarão nesse arquivo principal, pois é a que é comentada no artigo de entrega
- Dentro do diretório ``artigo`` tem o... artigo, que eu escrevi. No zip tem ele em .tex, mas deixei o pdf compilado também

# Como rodar (aka. "Como replicar os resultados alcançados no artigo de entrega")
- Instale os pacotes listados no ``requirements.txt``.
  - Eu instalei todos num ambiente conda -- maioria acaba sendo instalado via ``pip install unloth``, se não conseguir usar o requirements.
- Antes de começar a mexer em qualquer coisa, rode ``python leis_scrap.py``. O _script_ irá baixar as leis direto do _site_ do Planalto.gov, se os _links_ não tiverem quebrado. Esses dados são necessários pro treinamento
- Dentro do diretório ``rag``, rode nessa ordem:
  - ``python prepare_retrieval.py`` - _Script_ que prepara os índices para _retrieval_ RAG
  - ``python build_rag_dataset.py`` - Gera o _dataset_ que será usado no treinamento; para mais detalhes de como os dados se parecem, tem um exemplo no arquivo ``example.json`` - Vão ser uns 7000 exemplos desses (processo demorado, demorou 30h para mim numa rtx 4090)
  - ``python train_lora_rag.py`` - Depois de gerado o _dataset_, o treinamento pode começar executando esse _script_. Com 10 épocas, durou +- 10h para mim
# Inferência
- Tem 3 _scripts_ diferentes pra fazer inferência no diretório ``rag``: ,
  - ``infer_rag.py`` faz a inferência do modo normal, pede entrada e tudo mais. Nada de especial, é o padrão
  - ``infer_rag_cpu.py`` igual o acima mas faz a inferência na CPU. Não recomendo rodar, demora um monte e só deixa o computador lento. Fiz isso porque as GPUs que tinha acesso estavam ocupadas kk
  - ``infer_rag_force_ctx.py`` igual o padrão, mas você pode forçar um contexto específico dentro do código ao invés dele depender do _retrieval_. Não pensei em fazer o contexto ser _input_ na hora, então vai ter que editar o arquivo se quiser mexer o contexto

# Avaliação
- Essa parte não foi incluída no trabalho, deixei um _job_ rodando com ela mas no fim perdi os resultados. Basta rodar o ``eval_rag_pipeline.py``, leva umas 24h.
