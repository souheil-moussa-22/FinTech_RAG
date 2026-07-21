package com.finassistmini.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.finassistmini.model.GitRepository;
import com.finassistmini.model.RepositoryFile;
import com.finassistmini.repository.GitRepositoryRepository;
import com.finassistmini.repository.RepositoryFileRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.stereotype.Service;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

@Service
public class RepositorySummaryService {

    private static final Logger log = LoggerFactory.getLogger(RepositorySummaryService.class);
    private static final int MAX_CONTEXT_CHARS = 8_000;
    private static final int MAX_TREE_CONTEXT_CHARS = 3_000;
    private static final List<String> PRIORITY_FILES = List.of(
            "README.md", "README.rst", "README.txt", "readme.md",
            "pom.xml", "build.gradle", "build.gradle.kts", "package.json",
            "Cargo.toml", "go.mod", "requirements.txt", "setup.py",
            "docker-compose.yml", "docker-compose.yaml", "Dockerfile"
    );

    private final ChatModel                chatModel;
    private final GitRepositoryRepository  repositoryRepo;
    private final RepositoryFileRepository repositoryFileRepo;

    public RepositorySummaryService(ChatModel chatModel,
                                    GitRepositoryRepository repositoryRepo,
                                    RepositoryFileRepository repositoryFileRepo ) {
        this.chatModel      = chatModel;
        this.repositoryRepo = repositoryRepo;
        this.repositoryFileRepo = repositoryFileRepo;
    }

    public String getSummary(GitRepository repository) {
        if (repository.getSummary() != null && !repository.getSummary().isBlank()) {
            log.debug("Returning cached summary for repository {}", repository.getId());
            return extractOverviewOnly(repository.getSummary());
        }
        return generateAndCache(repository);
    }

    private String generateAndCache(GitRepository repository)  {
        log.info("Generating AI summary for repository: {}", repository.getName());

        String context = buildSummaryContext(repository);
        String prompt  = buildSummaryPrompt(repository.getName(), repository.getUrl(), context);

        try {
            String summary = chatModel.call(new Prompt(prompt))
                    .getResult()
                    .getOutput()
                    .getText();
            summary = extractOverviewOnly(summary);

            repository.setSummary(summary);
            repositoryRepo.save(repository);
            log.info("Summary generated and cached for repository {}", repository.getId());
            return summary;
        } catch (Exception e) {
            log.error("Failed to generate summary for repository {}: {}",
                    repository.getId(), e.getMessage());
            return "Summary generation failed: " + e.getMessage();
        }
    }

    private String buildSummaryContext(GitRepository repository) {
        Path repoRoot = Paths.get(repository.getLocalPath());
        StringBuilder context = new StringBuilder();
        List<RepositoryFile> files = repositoryFileRepo.findByRepositoryId(repository.getId());

        appendRepositoryMetadataContext(context, files);
        context.append("\nRepository file tree (from indexed metadata):\n")
                .append(getRepositoryFileTree(repository.getId()))
                .append("\n\n");

        for (String fileName : PRIORITY_FILES) {
            Path candidate = repoRoot.resolve(fileName);
            if (Files.exists(candidate) && Files.isRegularFile(candidate)) {
                try {
                    String content = Files.readString(candidate, StandardCharsets.UTF_8);
                    int limit = Math.min(content.length(), 2000);
                    context.append("=== ").append(fileName).append(" ===\n");
                    context.append(content, 0, limit);
                    if (content.length() > limit) context.append("\n[...truncated]");
                    context.append("\n\n");

                    if (context.length() >= MAX_CONTEXT_CHARS) break;
                } catch (IOException e) {
                    log.debug("Could not read priority file {}: {}", candidate, e.getMessage());
                }
            }
        }

        // If we have very little context, sample source files
        if (context.length() < 1000) {
            context.append(sampleSourceFiles(repoRoot));
        }

        return context.toString();
    }

    public String getRepositoryFileTree(Long repositoryId) {
        List<RepositoryFile> files = repositoryFileRepo.findByRepositoryId(repositoryId);
        if (files.isEmpty()) return "(no indexed file metadata available)";

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

        String result = tree.toString().trim();
        if (result.length() > MAX_TREE_CONTEXT_CHARS) {
            return result.substring(0, MAX_TREE_CONTEXT_CHARS) + "\n... (truncated)";
        }
        return result;
    }

    private void appendRepositoryMetadataContext(StringBuilder context, List<RepositoryFile> files) {
        if (files.isEmpty()) {
            context.append("Indexed repository metadata: none available.\n\n");
            return;
        }

        Map<String, Long> languageCounts = files.stream()
                .filter(f -> f.isIndexed() && f.getLanguage() != null && !f.getLanguage().isBlank())
                .collect(Collectors.groupingBy(RepositoryFile::getLanguage, TreeMap::new, Collectors.counting()));

        Map<String, Long> extensionCounts = files.stream()
                .map(RepositoryFile::getExtension)
                .filter(ext -> ext != null && !ext.isBlank())
                .collect(Collectors.groupingBy(ext -> ext.toLowerCase(), TreeMap::new, Collectors.counting()));

        long indexedCount = files.stream().filter(RepositoryFile::isIndexed).count();
        long skippedCount = files.stream().filter(RepositoryFile::isSkipped).count();
        long totalChunks = files.stream().map(RepositoryFile::getChunkCount).filter(c -> c != null).mapToLong(Integer::longValue).sum();

        context.append("Indexed repository metadata:\n")
                .append("- Total discovered files: ").append(files.size()).append('\n')
                .append("- Indexed files: ").append(indexedCount).append('\n')
                .append("- Skipped files: ").append(skippedCount).append('\n')
                .append("- Total chunks: ").append(totalChunks).append('\n')
                .append("- Languages (indexed): ").append(languageCounts.isEmpty() ? "unknown" : languageCounts).append('\n')
                .append("- Extensions: ").append(extensionCounts.isEmpty() ? "unknown" : extensionCounts).append("\n\n");
    }

    private String sampleSourceFiles(Path repoRoot) {
        StringBuilder sb = new StringBuilder();
        List<Path> sourceFiles = new ArrayList<>();

        try {
            Files.walk(repoRoot, 4)
                    .filter(Files::isRegularFile)
                    .filter(p -> {
                        String name = p.getFileName().toString();
                        return name.endsWith(".java") || name.endsWith(".py")
                                || name.endsWith(".ts") || name.endsWith(".go");
                    })
                    .limit(5)
                    .forEach(sourceFiles::add);

            for (Path file : sourceFiles) {
                String content = Files.readString(file, StandardCharsets.UTF_8);
                int limit = Math.min(content.length(), 800);
                sb.append("=== ").append(repoRoot.relativize(file)).append(" ===\n");
                sb.append(content, 0, limit).append("\n\n");
            }
        } catch (IOException e) {
            log.debug("Error sampling source files: {}", e.getMessage());
        }

        return sb.toString();
    }

    private String buildSummaryPrompt(String name, String url, String context) {
        return """
                You are a senior software architect. Analyze this Git repository and produce a concise technical overview.

                Repository: %s
                URL: %s

                Available files and content:
                %s

                Return markdown with exactly one section:
                Overview:
                - 3 to 6 concise technical bullet points about what this repository does.

                Do not include any other sections or headings.
                Do not include project purpose, main technologies, module/package structure, or file tree.
                The file tree is returned separately.
                """.formatted(name, url, context);
    }

    private String extractOverviewOnly(String rawSummary) {
        if (rawSummary == null) return "";
        String normalized = rawSummary.replace("\r\n", "\n").trim();
        if (normalized.isBlank()) return normalized;

        String[] stopHeadings = {
                "project purpose and problem solving",
                "main technologies and frameworks",
                "module and package structure",
                "architecture and design patterns",
                "build tool and configuration",
                "authentication and security approach",
                "database and persistence strategy",
                "public api and entry points",
                "key classes and their responsibilities",
                "notable design decisions and observations",
                "external dependencies",
                "folder structure understanding based on metadata"
        };

        String[] lines = normalized.split("\n");
        StringBuilder overview = new StringBuilder();
        boolean collecting = false;

        for (String line : lines) {
            String trimmed = line.trim();
            String normalizedHeading = trimmed
                    .replaceFirst("^#+\\s*", "")
                    .replaceFirst("^\\d+\\.\\s*", "")
                    .replaceAll(":\\s*$", "")
                    .toLowerCase();

            boolean isOverviewHeading = normalizedHeading.equals("overview");
            boolean isStopHeading = false;
            for (String stop : stopHeadings) {
                if (normalizedHeading.equals(stop)) {
                    isStopHeading = true;
                    break;
                }
            }

            if (isOverviewHeading) {
                if (overview.length() > 0) break;
                collecting = true;
                overview.append("Overview:\n");
                continue;
            }

            if (!collecting && overview.length() == 0 && !trimmed.isBlank()) {
                collecting = true;
                overview.append("Overview:\n");
            }

            if (collecting && isStopHeading) {
                break;
            }

            if (collecting) {
                overview.append(line).append('\n');
            }
        }

        String result = overview.toString().trim();
        return result.isBlank() ? "Overview:\n- Not available." : result;
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
                if (part.isBlank()) continue;
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