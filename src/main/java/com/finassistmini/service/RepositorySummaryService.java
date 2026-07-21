package com.finassistmini.service;

import com.finassistmini.model.GitRepository;
import com.finassistmini.model.RepositoryFile;
import com.finassistmini.model.RepositoryKnowledge;
import com.finassistmini.repository.GitRepositoryRepository;
import com.finassistmini.repository.RepositoryFileRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

@Service
public class RepositorySummaryService {

    private static final Logger log = LoggerFactory.getLogger(RepositorySummaryService.class);
    private static final int MAX_CONTEXT_CHARS = 12_000;

    private final ChatModel chatModel;
    private final GitRepositoryRepository repositoryRepo;
    private final RepositoryFileRepository repositoryFileRepo;
    private final RepositoryKnowledgeExtractor knowledgeExtractor;

    public RepositorySummaryService(ChatModel chatModel,
                                    GitRepositoryRepository repositoryRepo,
                                    RepositoryFileRepository repositoryFileRepo,
                                    RepositoryKnowledgeExtractor knowledgeExtractor) {
        this.chatModel = chatModel;
        this.repositoryRepo = repositoryRepo;
        this.repositoryFileRepo = repositoryFileRepo;
        this.knowledgeExtractor = knowledgeExtractor;
    }

    public String getSummary(GitRepository repository) {
        if (repository.getSummary() != null && !repository.getSummary().isBlank()) {
            log.debug("Returning cached summary for repository {}", repository.getId());
            return repository.getSummary();
        }
        return generateAndCache(repository);
    }

    public String getRepositoryFileTree(Long repositoryId) {
        List<RepositoryFile> files = repositoryFileRepo.findByRepositoryId(repositoryId);
        if (files.isEmpty()) {
            return "(no indexed file metadata available)";
        }

        Map<String, RepositoryFile> byPath = files.stream()
                .filter(f -> f.getPath() != null && !f.getPath().isBlank())
                .sorted((a, b) -> a.getPath().compareToIgnoreCase(b.getPath()))
                .collect(Collectors.toMap(
                        RepositoryFile::getPath,
                        f -> f,
                        (left, right) -> left,
                        TreeMap::new
                ));

        StringBuilder tree = new StringBuilder();
        TreeNode root = new TreeNode("");
        for (String path : byPath.keySet()) {
            root.insert(path);
        }
        root.render(tree, 0);

        return tree.toString().trim();
    }

    private String generateAndCache(GitRepository repository) {
        log.info("Generating AI summary for repository: {}", repository.getName());

        RepositoryKnowledge knowledge = knowledgeExtractor.extract(repository);
        String context = buildSummaryContext(knowledge);
        String prompt = buildSummaryPrompt(repository.getName(), repository.getUrl(), context);

        try {
            String summary = chatModel.call(new Prompt(prompt))
                    .getResult()
                    .getOutput()
                    .getText();

            repository.setSummary(summary);
            repositoryRepo.save(repository);
            log.info("Summary generated and cached for repository {}", repository.getId());
            return summary;
        } catch (Exception e) {
            log.error("Failed to generate summary for repository {}: {}", repository.getId(), e.getMessage());
            return "Summary generation failed: " + e.getMessage();
        }
    }

    private String buildSummaryContext(RepositoryKnowledge knowledge) {
        StringBuilder context = new StringBuilder();

        appendTextSection(context, "README", defaultIfBlank(knowledge.getReadme(), "Not available"));
        appendListSection(context, "Detected technologies", knowledge.getDetectedTechnologies());
        appendListSection(context, "Application entry points", knowledge.getEntryPoints());

        List<String> mainServices = new ArrayList<>();
        List<String> domainModel = new ArrayList<>();
        splitImportantClasses(knowledge.getImportantClasses(), mainServices, domainModel);

        appendListSection(context, "Main services", mainServices);
        appendListSection(context, "Domain model", domainModel);
        appendListSection(context, "Configuration", knowledge.getConfiguration());
        appendListSection(context, "Dependencies", knowledge.getDependencies());

        return truncateContext(context.toString());
    }

    private void splitImportantClasses(List<String> classes, List<String> mainServices, List<String> domainModel) {
        for (String value : classes) {
            String lower = value.toLowerCase();
            if (lower.contains("entity") || lower.contains("model") || lower.contains("dto")) {
                domainModel.add(value);
            } else {
                mainServices.add(value);
            }
        }
    }

    private void appendTextSection(StringBuilder target, String title, String content) {
        target.append(title)
                .append("\n------------------\n")
                .append(content)
                .append("\n\n");
    }

    private void appendListSection(StringBuilder target, String title, List<String> values) {
        target.append(title).append("\n------------------\n");
        if (values == null || values.isEmpty()) {
            target.append("- Not detected\n\n");
            return;
        }

        values.stream()
                .filter(value -> value != null && !value.isBlank())
                .limit(15)
                .forEach(value -> target.append("- ").append(value).append('\n'));

        target.append('\n');
    }

    private String truncateContext(String context) {
        if (context.length() <= MAX_CONTEXT_CHARS) {
            return context;
        }
        return context.substring(0, MAX_CONTEXT_CHARS) + "\n[...truncated]";
    }

    private String defaultIfBlank(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }

    private String buildSummaryPrompt(String name, String url, String context) {
        return """
                You are GitHub Copilot acting as a senior engineer.
                Analyze the project context and produce a concise product-level and technical overview.

                Project name: %s
                Project URL: %s

                Context:
                %s

                Your response must include exactly these sections:

                Overview
                - Explain what application this is.
                - Explain who it is built for.
                - Explain what users can do.
                - Explain what problem it solves.

                Stack
                - List languages.
                - List frameworks/platforms.
                - List databases and infrastructure.
                - Mention important libraries only when clearly supported by the context.

                Rules:
                - Do not mention repository statistics or metadata.
                - Do not mention indexing, chunking, embeddings, vector stores, or folder trees.
                - Do not speculate when the context is missing; state uncertainty briefly.
                - Keep the output concise and precise.
                """.formatted(name, url, context);
    }

    private static class TreeNode {
        private final String name;
        private final Map<String, TreeNode> children = new TreeMap<>(String.CASE_INSENSITIVE_ORDER);

        private TreeNode(String name) {
            this.name = name;
        }

        private void insert(String path) {
            String[] parts = path.split("/");
            TreeNode current = this;
            for (String part : parts) {
                if (part.isBlank()) {
                    continue;
                }
                current = current.children.computeIfAbsent(part, TreeNode::new);
            }
        }

        private void render(StringBuilder sb, int depth) {
            if (!name.isBlank()) {
                sb.append("  ".repeat(Math.max(0, depth - 1))).append("- ").append(name).append('\n');
            }
            for (TreeNode child : children.values()) {
                child.render(sb, depth + (name.isBlank() ? 0 : 1));
            }
        }
    }
}
