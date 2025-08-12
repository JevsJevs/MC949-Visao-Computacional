# MC949-Visao-Computacional

Template de projeto inspirado: Cookiecutter-data-science
https://cookiecutter-data-science.drivendata.org/#directory-structure

Divisão de pastas:
- data
    - interim: versoes dos dados originais, criar em subpastas nomeadas pelo padrão (stepDoTrab-iniciais-descricao) 
    - processed: versões final dos dados usados no canon do projeto (src)
    - raw: Dataset original em sua forma inalterada
- docs: documentação
    - subpastas referentes aos subescopos do projeto
- models: armazena modelos que serão criados
- notebooks: Playground de cada um para fins de pesquisa
    - criar conforme padrão de nomenclatura: stepDoTrab-initials-descricao
- src: Local do canone do projeto, ao final de pesquisa o código definitivo será posto aqui.


## Padrão de commit [Conventional commits]
https://www.conventionalcommits.org/en/v1.0.0/#summary 

### Instruções
- Usar os prefixos de type referentes ao tipo do commit (fig abaixo)
- Escopo    -> indicar o ID da task para rastreio no projeto github -> #1
- Mensagem  -> descrição da mudança 

Ex:  
**feat(#1): Tratamento incial dos dados**  
**docs(#2): explicação do algoritmo X**  
**refactor(#10): melhorando implementação do alg. Y**

![conventionalCommits](./docs/Project-Organization/conventionalCommit.png)