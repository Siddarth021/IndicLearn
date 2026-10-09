/**
 * IndicLearn — Student Dashboard & Visual Knowledge Map (Phase 8)
 * 
 * Core Architectural Principles:
 * 1. Strict VIEW layer: Does NOT calculate knowledge scores, mastery statuses,
 *    recommendation priorities, or adaptive item sequences.
 * 2. Communicates 'Estimated knowledge' and 'Evidence confidence', avoiding dogmatic certainty.
 * 3. Supports Multilingual rendering (English, Telugu, Romanized Telugu, Code-mix).
 * 4. Preserves grounded curriculum provenance and respects student isolation.
 */

(() => {
  'use strict';

  // Global Application State
  const state = {
    subjectId: 'signals_and_systems',
    studentId: 'student_demo_1',
    currentLanguage: 'en',
    graphData: null,
    selectedConceptId: null,
    activeSession: null,
    currentAssessmentItem: null,
    viewMode: 'graph', // 'graph' | 'list'
    // Graph Pan & Zoom transform
    pan: { x: 60, y: 40 },
    zoom: 1.0,
    isDragging: false,
    dragStart: { x: 0, y: 0 }
  };

  // DOM Elements Cache
  const el = {
    subjectSelect: document.getElementById('subjectSelect'),
    studentIdInput: document.getElementById('studentIdInput'),
    btnSwitchStudent: document.getElementById('btnSwitchStudent'),
    btnLoadDemo: document.getElementById('btnLoadDemo'),
    btnRefresh: document.getElementById('btnRefresh'),
    overviewStudentSubtitle: document.getElementById('overviewStudentSubtitle'),

    // Overview Metric Values
    valTrackedConcepts: document.getElementById('valTrackedConcepts'),
    valEstimatedKnowledge: document.getElementById('valEstimatedKnowledge'),
    valEvidenceConfidence: document.getElementById('valEvidenceConfidence'),
    valConceptsAssessed: document.getElementById('valConceptsAssessed'),
    valRecentActivity: document.getElementById('valRecentActivity'),
    pillMastered: document.getElementById('pillMastered'),
    pillDeveloping: document.getElementById('pillDeveloping'),
    pillEmerging: document.getElementById('pillEmerging'),
    pillUncertain: document.getElementById('pillUncertain'),
    pillUnknown: document.getElementById('pillUnknown'),

    // Knowledge Map & List
    mapStatusSummary: document.getElementById('mapStatusSummary'),
    btnViewGraph: document.getElementById('btnViewGraph'),
    btnViewList: document.getElementById('btnViewList'),
    btnZoomIn: document.getElementById('btnZoomIn'),
    btnZoomOut: document.getElementById('btnZoomOut'),
    btnZoomReset: document.getElementById('btnZoomReset'),
    graphContainer: document.getElementById('graphContainer'),
    svg: document.getElementById('knowledgeMapSvg'),
    viewport: document.getElementById('graphViewport'),
    edgeLayer: document.getElementById('edgeLayer'),
    edgeLabelLayer: document.getElementById('edgeLabelLayer'),
    nodeLayer: document.getElementById('nodeLayer'),
    conceptListView: document.getElementById('conceptListView'),
    conceptTableBody: document.getElementById('conceptTableBody'),

    // Concept Detail Panel
    detailPanel: document.getElementById('detailPanel'),
    detailConceptType: document.getElementById('detailConceptType'),
    detailEmptyState: document.getElementById('detailEmptyState'),
    detailBody: document.getElementById('detailBody'),
    detailConceptName: document.getElementById('detailConceptName'),
    detailConceptDesc: document.getElementById('detailConceptDesc'),
    detailPrereqList: document.getElementById('detailPrereqList'),
    detailDepList: document.getElementById('detailDepList'),
    detailKnowledgeScore: document.getElementById('detailKnowledgeScore'),
    detailKnowledgeBar: document.getElementById('detailKnowledgeBar'),
    detailConfidence: document.getElementById('detailConfidence'),
    detailConfidenceBar: document.getElementById('detailConfidenceBar'),
    detailMasteryBadge: document.getElementById('detailMasteryBadge'),
    detailEvidenceCount: document.getElementById('detailEvidenceCount'),
    detailStrongEvidenceCount: document.getElementById('detailStrongEvidenceCount'),
    detailWeakEvidenceCount: document.getElementById('detailWeakEvidenceCount'),
    evidenceAccordionToggle: document.getElementById('evidenceAccordionToggle'),
    evidenceAccordionBody: document.getElementById('evidenceAccordionBody'),
    detailExplanationReason: document.getElementById('detailExplanationReason'),
    detailEvidenceTimeline: document.getElementById('detailEvidenceTimeline'),
    btnInspectSource: document.getElementById('btnInspectSource'),
    btnStartConceptAssessment: document.getElementById('btnStartConceptAssessment'),

    // Recommendations
    recommendationsContainer: document.getElementById('recommendationsContainer'),

    // Assessment Modal
    assessmentModal: document.getElementById('assessmentModal'),
    btnCloseAssessment: document.getElementById('btnCloseAssessment'),
    assessmentSubtitle: document.getElementById('assessmentSubtitle'),
    assessmentItemContainer: document.getElementById('assessmentItemContainer'),
    assessmentItemDifficulty: document.getElementById('assessmentItemDifficulty'),
    assessmentItemConcept: document.getElementById('assessmentItemConcept'),
    assessmentLangSelect: document.getElementById('assessmentLangSelect'),
    assessmentQuestionText: document.getElementById('assessmentQuestionText'),
    assessmentAnswerForm: document.getElementById('assessmentAnswerForm'),
    assessmentOptionsList: document.getElementById('assessmentOptionsList'),
    btnSubmitAssessmentAnswer: document.getElementById('btnSubmitAssessmentAnswer'),
    assessmentResultContainer: document.getElementById('assessmentResultContainer'),
    assessmentResultBanner: document.getElementById('assessmentResultBanner'),
    assessmentResultIcon: document.getElementById('assessmentResultIcon'),
    assessmentResultTitle: document.getElementById('assessmentResultTitle'),
    assessmentResultFeedback: document.getElementById('assessmentResultFeedback'),
    assessmentUpdatedScore: document.getElementById('assessmentUpdatedScore'),
    assessmentUpdatedScoreBar: document.getElementById('assessmentUpdatedScoreBar'),
    assessmentUpdatedConfidence: document.getElementById('assessmentUpdatedConfidence'),
    assessmentUpdatedConfidenceBar: document.getElementById('assessmentUpdatedConfidenceBar'),
    assessmentUpdatedMastery: document.getElementById('assessmentUpdatedMastery'),
    btnNextAssessmentItem: document.getElementById('btnNextAssessmentItem'),
    btnFinishAssessment: document.getElementById('btnFinishAssessment'),

    // Source Modal
    sourceModal: document.getElementById('sourceModal'),
    btnCloseSource: document.getElementById('btnCloseSource'),
    sourceDocId: document.getElementById('sourceDocId'),
    sourcePageSection: document.getElementById('sourcePageSection'),
    sourceChunkId: document.getElementById('sourceChunkId'),
    sourceChunkText: document.getElementById('sourceChunkText'),

    // Upload Modal
    btnOpenUploadModal: document.getElementById('btnOpenUploadModal'),
    uploadModal: document.getElementById('uploadModal'),
    btnCloseUpload: document.getElementById('btnCloseUpload'),
    uploadFileInput: document.getElementById('uploadFileInput'),
    uploadSubjectInput: document.getElementById('uploadSubjectInput'),
    uploadStatusBox: document.getElementById('uploadStatusBox'),
    uploadStatusText: document.getElementById('uploadStatusText'),
    btnSubmitUpload: document.getElementById('btnSubmitUpload'),

    // Toast
    toastNotification: document.getElementById('toastNotification')
  };

  // =========================================================================
  // API Client Methods (Strict View - Calls Backend Endpoints)
  // =========================================================================

  const api = {
    async fetchSubjects() {
      const res = await fetch('/api/subjects');
      if (!res.ok) throw new Error('Failed to load subjects');
      return res.json();
    },

    async fetchDashboardOverview(studentId, subjectId) {
      const res = await fetch(`/api/students/${encodeURIComponent(studentId)}/dashboard?subject_id=${encodeURIComponent(subjectId)}`);
      if (!res.ok) throw new Error('Failed to load dashboard metrics');
      return res.json();
    },

    async fetchKnowledgeMap(studentId, subjectId) {
      const res = await fetch(`/api/subjects/${encodeURIComponent(subjectId)}/graph?student_id=${encodeURIComponent(studentId)}`);
      if (!res.ok) throw new Error('Failed to load knowledge map');
      return res.json();
    },

    async fetchConceptDetails(studentId, conceptId, subjectId) {
      const res = await fetch(`/api/concepts/${encodeURIComponent(conceptId)}?student_id=${encodeURIComponent(studentId)}&subject_id=${encodeURIComponent(subjectId)}`);
      if (!res.ok) throw new Error('Failed to load concept details');
      return res.json();
    },

    async fetchRecommendations(studentId, subjectId, topK = 6) {
      const res = await fetch(`/api/students/${encodeURIComponent(studentId)}/recommendations?subject_id=${encodeURIComponent(subjectId)}&top_k=${topK}`);
      if (!res.ok) throw new Error('Failed to load recommendations');
      return res.json();
    },

    async startAssessment(studentId, subjectId, conceptId = null, lang = 'en') {
      const payload = {
        student_id: studentId,
        subject_id: subjectId,
        concept_id: conceptId,
        requested_language: lang
      };
      const res = await fetch('/api/assessment/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error('Failed to initiate assessment session');
      return res.json();
    },

    async submitAssessmentResponse(sessionId, itemId, answer, lang = 'en') {
      const payload = {
        session_id: sessionId,
        item_id: itemId,
        selected_answer: answer,
        requested_language: lang
      };
      const res = await fetch(`/api/assessment/sessions/${encodeURIComponent(sessionId)}/responses`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error('Failed to submit response');
      return res.json();
    },

    async fetchChunk(chunkId) {
      const res = await fetch(`/api/chunks/${encodeURIComponent(chunkId)}`);
      if (!res.ok) return null;
      return res.json();
    },

    async loadDemoData(studentId) {
      const res = await fetch(`/api/demo/load?student_id=${encodeURIComponent(studentId)}`, {
        method: 'POST'
      });
      if (!res.ok) throw new Error('Failed to load demo fixture data');
      return res.json();
    }
  };

  // =========================================================================
  // Toast & Notifications
  // =========================================================================

  function showToast(message, duration = 3500) {
    el.toastNotification.textContent = message;
    el.toastNotification.classList.remove('hidden');
    setTimeout(() => {
      el.toastNotification.classList.add('hidden');
    }, duration);
  }

  // =========================================================================
  // Subject & Topic Synchronization
  // =========================================================================

  async function populateSubjects() {
    try {
      const subjects = await api.fetchSubjects();
      if (!Array.isArray(subjects) || subjects.length === 0) return;

      el.subjectSelect.innerHTML = '';
      for (const sub of subjects) {
        const opt = document.createElement('option');
        opt.value = sub.subject_id;
        const count = sub.concept_count !== undefined ? sub.concept_count : 0;
        opt.textContent = `${sub.name} (${count} concepts)`;
        if (sub.subject_id === state.subjectId) {
          opt.selected = true;
        }
        el.subjectSelect.appendChild(opt);
      }

      if (!subjects.some(s => s.subject_id === state.subjectId)) {
        state.subjectId = subjects[0].subject_id;
        el.subjectSelect.value = state.subjectId;
      }
    } catch (err) {
      console.warn('Could not populate subjects list:', err);
    }
  }

  // =========================================================================
  // Dashboard & Metrics Rendering
  // =========================================================================

  async function loadDashboard() {
    el.overviewStudentSubtitle.textContent = `Viewing student: ${state.studentId}`;
    try {
      // 1. Load Overview Metrics
      const overview = await api.fetchDashboardOverview(state.studentId, state.subjectId);
      renderOverview(overview);

      // 2. Load Knowledge Map
      const mapData = await api.fetchKnowledgeMap(state.studentId, state.subjectId);
      state.graphData = mapData;
      renderKnowledgeMap(mapData);
      renderConceptList(mapData);

      // 3. Load Recommendations
      const recs = await api.fetchRecommendations(state.studentId, state.subjectId);
      renderRecommendations(recs);

      // 4. Re-select concept if already chosen
      if (state.selectedConceptId) {
        selectConcept(state.selectedConceptId);
      }
    } catch (err) {
      console.error(err);
      showToast(`Error updating dashboard: ${err.message}`);
    }
  }

  function renderOverview(data) {
    el.valTrackedConcepts.textContent = data.total_concepts_tracked;
    el.valEstimatedKnowledge.textContent = `${(data.mean_knowledge_score * 100).toFixed(1)}%`;
    el.valEvidenceConfidence.textContent = `${(data.mean_confidence * 100).toFixed(1)}%`;
    el.valConceptsAssessed.textContent = `${data.concepts_assessed} / ${data.total_concepts_tracked}`;
    el.valRecentActivity.textContent = data.recent_activity_count;

    el.pillMastered.textContent = `M: ${data.mastered_count}`;
    el.pillDeveloping.textContent = `D: ${data.developing_count}`;
    el.pillEmerging.textContent = `E: ${data.emerging_count}`;
    el.pillUncertain.textContent = `U: ${data.uncertain_count}`;
    el.pillUnknown.textContent = `?: ${data.unknown_count}`;
  }

  // =========================================================================
  // 2D Knowledge Map Visualization (SVG with Pan, Zoom & Auto Layout)
  // =========================================================================

  function computeLayout(nodes, edges) {
    // Topological depth layout: compute in-degree of PREREQUISITE relations
    const inDegree = {};
    const adj = {};
    nodes.forEach(n => {
      inDegree[n.id] = 0;
      adj[n.id] = [];
    });

    edges.forEach(e => {
      if (e.relation_type === 'PREREQUISITE_OF' && inDegree[e.target] !== undefined) {
        inDegree[e.target] += 1;
        adj[e.source].push(e.target);
      }
    });

    // Compute layer for each node (longest path from sources)
    const layers = {};
    nodes.forEach(n => {
      layers[n.id] = 0;
    });

    // Relax layer depths
    for (let iter = 0; iter < 10; iter++) {
      edges.forEach(e => {
        if (e.relation_type === 'PREREQUISITE_OF' && layers[e.source] !== undefined && layers[e.target] !== undefined) {
          if (layers[e.target] <= layers[e.source]) {
            layers[e.target] = layers[e.source] + 1;
          }
        }
      });
    }

    // Group nodes by layer
    const layerBuckets = {};
    nodes.forEach(n => {
      const l = layers[n.id] || 0;
      if (!layerBuckets[l]) layerBuckets[l] = [];
      layerBuckets[l].push(n);
    });

    // Assign X, Y coordinates
    const layerKeys = Object.keys(layerBuckets).map(Number).sort((a, b) => a - b);
    const nodeCoords = {};
    const colSpacing = 220;
    const rowSpacing = 95;

    layerKeys.forEach(l => {
      const bucket = layerBuckets[l];
      const startY = 80 + (bucket.length === 1 ? 90 : 0);
      bucket.forEach((node, idx) => {
        nodeCoords[node.id] = {
          x: 90 + l * colSpacing,
          y: startY + idx * rowSpacing
        };
      });
    });

    return nodeCoords;
  }

  function getStatusColor(status) {
    switch (status) {
      case 'MASTERED': return 'var(--status-mastered)';
      case 'DEVELOPING': return 'var(--status-developing)';
      case 'EMERGING': return 'var(--status-emerging)';
      case 'UNCERTAIN': return 'var(--status-uncertain)';
      default: return 'var(--status-unknown)';
    }
  }

  function renderKnowledgeMap(data) {
    el.mapStatusSummary.textContent = `${data.nodes.length} concepts, ${data.edges.length} relations`;

    // Clear existing SVG elements
    el.edgeLayer.innerHTML = '';
    el.edgeLabelLayer.innerHTML = '';
    el.nodeLayer.innerHTML = '';

    const coords = computeLayout(data.nodes, data.edges);

    // 1. Render Edges
    data.edges.forEach(edge => {
      const src = coords[edge.source];
      const tgt = coords[edge.target];
      if (!src || !tgt) return;

      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      const dx = tgt.x - src.x;
      const dy = tgt.y - src.y;
      const cx1 = src.x + dx * 0.5;
      const cy1 = src.y;
      const cx2 = src.x + dx * 0.5;
      const cy2 = tgt.y;

      const d = `M ${src.x} ${src.y} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${tgt.x} ${tgt.y}`;
      path.setAttribute('d', d);
      path.setAttribute('class', `edge-line edge-${edge.relation_type.toLowerCase()}`);
      path.setAttribute('fill', 'none');

      // Set directional arrow marker
      if (edge.relation_type === 'PREREQUISITE_OF') {
        path.setAttribute('marker-end', 'url(#arrow-prereq)');
      } else if (edge.relation_type === 'PART_OF') {
        path.setAttribute('marker-end', 'url(#arrow-partof)');
      } else {
        path.setAttribute('marker-end', 'url(#arrow-related)');
      }

      el.edgeLayer.appendChild(path);

      // Label at mid-curve
      const labelText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      labelText.setAttribute('x', (src.x + tgt.x) / 2);
      labelText.setAttribute('y', (src.y + tgt.y) / 2 - 4);
      labelText.setAttribute('class', 'edge-label');
      labelText.setAttribute('text-anchor', 'middle');
      labelText.textContent = edge.label;
      el.edgeLabelLayer.appendChild(labelText);
    });

    // 2. Render Nodes
    data.nodes.forEach(node => {
      const pos = coords[node.id];
      if (!pos) return;

      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute('class', `node-group ${state.selectedConceptId === node.id ? 'selected' : ''}`);
      g.setAttribute('id', `node_${node.id}`);
      g.setAttribute('transform', `translate(${pos.x}, ${pos.y})`);
      g.setAttribute('role', 'button');
      g.setAttribute('aria-label', `${node.name}: Status ${node.mastery_status}`);

      // Selection halo (visible when selected)
      const halo = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      halo.setAttribute('class', 'node-halo');
      halo.setAttribute('r', '27');
      halo.setAttribute('fill', 'none');
      halo.setAttribute('stroke', 'transparent');
      g.appendChild(halo);

      // Outer status ring
      const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      ring.setAttribute('r', '22');
      ring.setAttribute('fill', '#111827');
      ring.setAttribute('stroke', getStatusColor(node.mastery_status));
      ring.setAttribute('stroke-width', '3');
      ring.setAttribute('class', 'node-ring');
      g.appendChild(ring);

      // Inner core circle
      const core = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      core.setAttribute('r', '18');
      core.setAttribute('fill', '#1e293b');
      core.setAttribute('class', 'node-core');
      g.appendChild(core);

      // Node label (Concept canonical name)
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('class', 'node-label');
      text.setAttribute('x', '0');
      text.setAttribute('y', '36');
      text.setAttribute('text-anchor', 'middle');
      // Truncate if long
      const displayName = node.name.length > 24 ? node.name.slice(0, 22) + '…' : node.name;
      text.textContent = displayName;
      g.appendChild(text);

      // Score / confidence indicator inside node
      const scoreText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      scoreText.setAttribute('class', 'node-score-text');
      scoreText.setAttribute('x', '0');
      scoreText.setAttribute('y', '3');
      scoreText.setAttribute('text-anchor', 'middle');
      if (node.evidence_count > 0) {
        scoreText.textContent = `${Math.round(node.knowledge_score * 100)}%`;
      } else {
        scoreText.textContent = '?';
      }
      g.appendChild(scoreText);

      // Event listener for node selection
      g.addEventListener('click', (evt) => {
        evt.stopPropagation();
        selectConcept(node.id);
      });

      el.nodeLayer.appendChild(g);
    });

    updateViewportTransform();
  }

  function updateViewportTransform() {
    el.viewport.setAttribute(
      'transform',
      `translate(${state.pan.x}, ${state.pan.y}) scale(${state.zoom})`
    );
  }

  // Accessible Concept Table View
  function renderConceptList(data) {
    el.conceptTableBody.innerHTML = '';
    data.nodes.forEach(node => {
      const tr = document.createElement('tr');
      const scoreStr = node.evidence_count > 0 ? `${(node.knowledge_score * 100).toFixed(1)}%` : 'No evidence';
      const confStr = node.evidence_count > 0 ? `${(node.confidence * 100).toFixed(1)}%` : '0.0%';

      tr.innerHTML = `
        <td><strong>${node.name}</strong></td>
        <td><span class="badge-type">${node.type}</span></td>
        <td>${scoreStr}</td>
        <td>${confStr}</td>
        <td><span class="status-badge status-${node.mastery_status.toLowerCase()}">${node.mastery_status}</span></td>
        <td><button class="btn btn-secondary btn-sm" data-cid="${node.id}">Inspect</button></td>
      `;

      tr.querySelector('button').addEventListener('click', () => {
        selectConcept(node.id);
      });
      el.conceptTableBody.appendChild(tr);
    });
  }

  // =========================================================================
  // Concept Details & Evidence Transparency Drawer
  // =========================================================================

  async function selectConcept(conceptId) {
    state.selectedConceptId = conceptId;

    // Highlight node in SVG
    document.querySelectorAll('.node-group').forEach(ng => ng.classList.remove('selected'));
    const selectedEl = document.getElementById(`node_${conceptId}`);
    if (selectedEl) selectedEl.classList.add('selected');

    try {
      const details = await api.fetchConceptDetails(state.studentId, conceptId, state.subjectId);
      renderConceptDetails(details);
    } catch (err) {
      console.error(err);
      showToast(`Failed to load concept details: ${err.message}`);
    }
  }

  function renderConceptDetails(d) {
    el.detailEmptyState.classList.add('hidden');
    el.detailBody.classList.remove('hidden');

    el.detailConceptType.textContent = d.concept_type;
    el.detailConceptName.textContent = d.canonical_name;
    el.detailConceptDesc.textContent = d.description || 'Curriculum concept in Signals and Systems.';

    // Prerequisites
    el.detailPrereqList.innerHTML = '';
    if (d.prerequisites.length > 0) {
      d.prerequisites.forEach(p => {
        const span = document.createElement('span');
        span.className = 'tag-item';
        span.textContent = p.name;
        span.title = 'Click to focus prerequisite';
        span.addEventListener('click', () => selectConcept(p.id));
        el.detailPrereqList.appendChild(span);
      });
    } else {
      el.detailPrereqList.innerHTML = '<span class="text-muted">None specified</span>';
    }

    // Dependents
    el.detailDepList.innerHTML = '';
    if (d.dependents.length > 0) {
      d.dependents.forEach(dep => {
        const span = document.createElement('span');
        span.className = 'tag-item';
        span.textContent = dep.name;
        span.title = 'Click to focus dependent concept';
        span.addEventListener('click', () => selectConcept(dep.id));
        el.detailDepList.appendChild(span);
      });
    } else {
      el.detailDepList.innerHTML = '<span class="text-muted">None specified</span>';
    }

    // Student Knowledge Overlay
    const scorePct = Math.round(d.knowledge_score * 100);
    const confPct = Math.round(d.confidence * 100);
    el.detailKnowledgeScore.textContent = `${(d.knowledge_score).toFixed(2)} (${scorePct}%)`;
    el.detailKnowledgeBar.style.width = `${scorePct}%`;
    el.detailConfidence.textContent = `${(d.confidence).toFixed(2)} (${confPct}%)`;
    el.detailConfidenceBar.style.width = `${confPct}%`;

    el.detailMasteryBadge.textContent = d.mastery_status;
    el.detailMasteryBadge.className = `status-badge status-${d.mastery_status.toLowerCase()}`;

    el.detailEvidenceCount.textContent = d.evidence_count;
    el.detailStrongEvidenceCount.textContent = d.strong_evidence_count;
    el.detailWeakEvidenceCount.textContent = d.weak_evidence_count;

    // Evidence Transparency Explanation
    el.detailExplanationReason.textContent = d.explanation_reason;
    el.detailEvidenceTimeline.innerHTML = '';

    if (d.evidence_history && d.evidence_history.length > 0) {
      d.evidence_history.forEach(ev => {
        const item = document.createElement('div');
        item.className = 'timeline-item';
        let iconHtml = '<span class="timeline-icon interaction">•</span>';
        if (ev.event_type.includes('CORRECT')) {
          iconHtml = '<span class="timeline-icon correct">✓</span>';
        } else if (ev.event_type.includes('INCORRECT')) {
          iconHtml = '<span class="timeline-icon incorrect">✗</span>';
        }

        const dateStr = new Date(ev.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        item.innerHTML = `
          ${iconHtml}
          <div style="flex:1;">
            <strong>${ev.event_type}</strong> (${ev.strength}): ${ev.summary}
          </div>
          <span class="text-muted">${dateStr}</span>
        `;
        el.detailEvidenceTimeline.appendChild(item);
      });
    } else {
      el.detailEvidenceTimeline.innerHTML = '<span class="text-muted">No evidence events recorded yet.</span>';
    }

    // Source Inspector Button
    el.btnInspectSource.onclick = () => {
      if (d.source_chunk_ids && d.source_chunk_ids.length > 0) {
        openSourceModal(d.source_chunk_ids[0]);
      } else {
        openSourceModal(null, d.canonical_name);
      }
    };

    // Assessment Launch Button
    el.btnStartConceptAssessment.onclick = () => {
      launchAssessment(d.concept_id);
    };
  }

  // =========================================================================
  // Recommendations Section
  // =========================================================================

  function renderRecommendations(recs) {
    el.recommendationsContainer.innerHTML = '';

    if (!recs || recs.length === 0) {
      el.recommendationsContainer.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="empty-icon">🎉</div>
          <p>No immediate remediation actions flagged. Keep exploring the knowledge map or test your mastery.</p>
        </div>
      `;
      return;
    }

    recs.forEach(rec => {
      const card = document.createElement('div');
      card.className = 'rec-card';

      const reasonsHtml = rec.explanation
        ? `<div class="rec-reason-item"><span class="rec-reason-bullet">•</span><span>${rec.explanation}</span></div>`
        : rec.reason_codes.map(r => `<div class="rec-reason-item"><span class="rec-reason-bullet">•</span><span>${r}</span></div>`).join('');

      card.innerHTML = `
        <div class="rec-header">
          <div>
            <h3 class="rec-concept-name">${rec.canonical_name}</h3>
            <span class="rec-priority">Priority: ${rec.priority_score.toFixed(2)}</span>
          </div>
          <span class="rec-action-badge">${rec.action_type}</span>
        </div>

        <div class="rec-reasons">
          ${reasonsHtml}
        </div>

        <div class="rec-meta-row">
          <span>Est. Score: <strong>${Math.round(rec.knowledge_score * 100)}%</strong></span>
          <span>Conf: <strong>${Math.round(rec.confidence * 100)}%</strong></span>
          <span class="status-badge status-${rec.mastery_status.toLowerCase()}">${rec.mastery_status}</span>
        </div>

        <div class="rec-actions">
          <button class="btn btn-primary btn-sm btn-block btn-rec-action" data-action="${rec.action_type}" data-cid="${rec.concept_id}">
            ${rec.action_type === 'TAKE_ASSESSMENT' ? '🎯 Take Assessment' : '📖 Review Concept'}
          </button>
          <button class="btn btn-secondary btn-sm btn-rec-view" data-cid="${rec.concept_id}">
            Map
          </button>
        </div>
      `;

      card.querySelector('.btn-rec-action').addEventListener('click', () => {
        if (rec.action_type === 'TAKE_ASSESSMENT') {
          launchAssessment(rec.concept_id);
        } else {
          selectConcept(rec.concept_id);
          if (rec.source_chunk_ids && rec.source_chunk_ids.length > 0) {
            openSourceModal(rec.source_chunk_ids[0]);
          } else {
            openSourceModal(null, rec.canonical_name);
          }
        }
      });

      card.querySelector('.btn-rec-view').addEventListener('click', () => {
        selectConcept(rec.concept_id);
        // Center pan on selected node
        const nodeEl = document.getElementById(`node_${rec.concept_id}`);
        if (nodeEl) {
          nodeEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      });

      el.recommendationsContainer.appendChild(card);
    });
  }

  // =========================================================================
  // Adaptive Assessment Integration (Delegates strictly to Phase 7A backend)
  // =========================================================================

  async function launchAssessment(conceptId = null) {
    try {
      showToast('Initiating adaptive assessment session...');
      const sessionView = await api.startAssessment(
        state.studentId,
        state.subjectId,
        conceptId,
        state.currentLanguage
      );

      state.activeSession = sessionView;
      openAssessmentModal(sessionView);
    } catch (err) {
      console.error(err);
      showToast(`Cannot start assessment: ${err.message}`);
    }
  }

  function openAssessmentModal(sessionView) {
    el.assessmentModal.classList.remove('hidden');
    el.assessmentResultContainer.classList.add('hidden');
    el.assessmentItemContainer.classList.remove('hidden');

    if (!sessionView.current_item) {
      el.assessmentQuestionText.textContent = 'No eligible assessment items available for this concept/session.';
      el.assessmentOptionsList.innerHTML = '';
      el.btnSubmitAssessmentAnswer.classList.add('hidden');
      return;
    }

    renderAssessmentItem(sessionView.current_item, sessionView.current_index);
  }

  function renderAssessmentItem(item, itemIndex) {
    state.currentAssessmentItem = item;
    el.assessmentSubtitle.textContent = `Question ${itemIndex + 1} (Adaptive Selection)`;
    el.assessmentItemDifficulty.textContent = item.difficulty;
    el.assessmentItemConcept.textContent = item.concept_id;
    el.assessmentQuestionText.textContent = item.question;
    el.btnSubmitAssessmentAnswer.classList.remove('hidden');

    el.assessmentOptionsList.innerHTML = '';
    if (item.options && item.options.length > 0) {
      item.options.forEach((opt, idx) => {
        const label = document.createElement('label');
        label.className = 'option-label';
        const optId = `opt_${idx}`;
        label.innerHTML = `
          <input type="radio" name="assessmentAnswer" id="${optId}" value="${opt}" />
          <span>${opt}</span>
        `;
        el.assessmentOptionsList.appendChild(label);
      });
    } else {
      // Freeform or True/False fallback
      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'form-input btn-block';
      input.name = 'assessmentAnswer';
      input.placeholder = 'Type your answer here...';
      el.assessmentOptionsList.appendChild(input);
    }
  }

  async function submitAssessmentAnswer() {
    if (!state.activeSession || !state.currentAssessmentItem) return;

    let selectedAnswer = null;
    const radioChecked = document.querySelector('input[name="assessmentAnswer"]:checked');
    if (radioChecked) {
      selectedAnswer = radioChecked.value;
    } else {
      const textInput = document.querySelector('input[name="assessmentAnswer"][type="text"]');
      if (textInput && textInput.value.trim()) {
        selectedAnswer = textInput.value.trim();
      }
    }

    if (!selectedAnswer) {
      showToast('Please select or enter an answer before submitting.');
      return;
    }

    try {
      el.btnSubmitAssessmentAnswer.disabled = true;
      el.btnSubmitAssessmentAnswer.textContent = 'Evaluating...';

      const result = await api.submitAssessmentResponse(
        state.activeSession.session_id,
        state.currentAssessmentItem.item_id,
        selectedAnswer,
        state.currentLanguage
      );

      renderAssessmentResult(result);
    } catch (err) {
      console.error(err);
      showToast(`Error submitting response: ${err.message}`);
    } finally {
      el.btnSubmitAssessmentAnswer.disabled = false;
      el.btnSubmitAssessmentAnswer.textContent = 'Submit Answer';
    }
  }

  function renderAssessmentResult(res) {
    el.assessmentItemContainer.classList.add('hidden');
    el.assessmentResultContainer.classList.remove('hidden');

    if (res.correct) {
      el.assessmentResultBanner.className = 'result-banner correct';
      el.assessmentResultIcon.textContent = '✓';
      el.assessmentResultTitle.textContent = 'Demonstrated Understanding!';
    } else {
      el.assessmentResultBanner.className = 'result-banner incorrect';
      el.assessmentResultIcon.textContent = '✗';
      el.assessmentResultTitle.textContent = 'Needs Further Review';
    }

    el.assessmentResultFeedback.textContent = res.feedback;

    const scorePct = Math.round(res.updated_knowledge_score * 100);
    const confPct = Math.round(res.updated_confidence * 100);
    el.assessmentUpdatedScore.textContent = `${res.updated_knowledge_score.toFixed(2)} (${scorePct}%)`;
    el.assessmentUpdatedScoreBar.style.width = `${scorePct}%`;
    el.assessmentUpdatedConfidence.textContent = `${res.updated_confidence.toFixed(2)} (${confPct}%)`;
    el.assessmentUpdatedConfidenceBar.style.width = `${confPct}%`;

    el.assessmentUpdatedMastery.textContent = res.updated_mastery_status;
    el.assessmentUpdatedMastery.className = `status-badge status-${res.updated_mastery_status.toLowerCase()}`;

    if (res.session_completed || !res.next_item) {
      el.btnNextAssessmentItem.classList.add('hidden');
      el.btnFinishAssessment.textContent = 'Session Complete ➔ View Dashboard';
    } else {
      el.btnNextAssessmentItem.classList.remove('hidden');
      el.btnNextAssessmentItem.onclick = () => {
        el.assessmentResultContainer.classList.add('hidden');
        el.assessmentItemContainer.classList.remove('hidden');
        state.activeSession.current_index += 1;
        renderAssessmentItem(res.next_item, state.activeSession.current_index);
      };
    }
  }

  function closeAssessmentModal() {
    el.assessmentModal.classList.add('hidden');
    state.activeSession = null;
    state.currentAssessmentItem = null;
    // Always refresh student knowledge state and dashboard on session close
    loadDashboard();
  }

  // =========================================================================
  // Source Material Inspector Modal
  // =========================================================================

  async function openSourceModal(chunkId, conceptName = 'Concept') {
    el.sourceModal.classList.remove('hidden');
    if (!chunkId) {
      el.sourceDocId.textContent = 'None';
      el.sourcePageSection.textContent = 'N/A';
      el.sourceChunkId.textContent = 'N/A';
      el.sourceChunkText.innerHTML = `
        <div style="color: var(--status-emerging); margin-bottom: 0.5rem;">⚠️ Source unavailable</div>
        <p>No primary textbook chunk or document citation is indexed for "${conceptName}".</p>
      `;
      return;
    }

    try {
      el.sourceChunkText.textContent = 'Loading source curriculum text...';
      const chunk = await api.fetchChunk(chunkId);
      if (!chunk) {
        el.sourceDocId.textContent = 'Unknown';
        el.sourcePageSection.textContent = 'N/A';
        el.sourceChunkId.textContent = chunkId;
        el.sourceChunkText.innerHTML = `<span style="color: var(--status-emerging);">Source unavailable</span>: Chunk ID ${chunkId} not found in curriculum corpus.`;
        return;
      }

      el.sourceDocId.textContent = chunk.document_id;
      const pageSec = [
        chunk.page_number ? `Page ${chunk.page_number}` : null,
        chunk.section ? `Section: ${chunk.section}` : null
      ].filter(Boolean).join(' | ') || 'Curriculum Reference';
      el.sourcePageSection.textContent = pageSec;
      el.sourceChunkId.textContent = chunk.chunk_id;
      el.sourceChunkText.textContent = chunk.text;
    } catch (err) {
      el.sourceChunkText.textContent = `Error loading source material: ${err.message}`;
    }
  }

  // =========================================================================
  // Event Listeners & Interactive Handlers
  // =========================================================================

  function setupEventListeners() {
    // Topic / Subject Switcher
    el.subjectSelect.addEventListener('change', (e) => {
      state.subjectId = e.target.value;
      state.selectedConceptId = null;
      loadDashboard();
      const selectedName = el.subjectSelect.options[el.subjectSelect.selectedIndex]?.text || state.subjectId;
      showToast(`Switched active topic to: ${selectedName}`);
    });

    // Student Switcher
    el.btnSwitchStudent.addEventListener('click', () => {
      const val = el.studentIdInput.value.trim();
      if (val) {
        state.studentId = val;
        state.selectedConceptId = null;
        loadDashboard();
        showToast(`Switched active view to student: ${state.studentId}`);
      }
    });

    // Refresh Button
    el.btnRefresh.addEventListener('click', () => {
      loadDashboard();
      showToast('Dashboard reloaded from backend state.');
    });

    // Load Demo Data Button
    el.btnLoadDemo.addEventListener('click', async () => {
      try {
        el.btnLoadDemo.disabled = true;
        const res = await api.loadDemoData(state.studentId);
        showToast(`[DEMO DATA ONLY] Loaded sample events for ${state.studentId}`);
        await loadDashboard();
      } catch (err) {
        showToast(`Error loading demo fixture: ${err.message}`);
      } finally {
        el.btnLoadDemo.disabled = false;
      }
    });

    // View Mode Toggle (Graph vs List)
    el.btnViewGraph.addEventListener('click', () => {
      state.viewMode = 'graph';
      el.btnViewGraph.classList.add('active');
      el.btnViewList.classList.remove('active');
      el.btnViewGraph.setAttribute('aria-pressed', 'true');
      el.btnViewList.setAttribute('aria-pressed', 'false');
      el.graphContainer.classList.remove('hidden');
      el.conceptListView.classList.add('hidden');
    });

    el.btnViewList.addEventListener('click', () => {
      state.viewMode = 'list';
      el.btnViewList.classList.add('active');
      el.btnViewGraph.classList.remove('active');
      el.btnViewList.setAttribute('aria-pressed', 'true');
      el.btnViewGraph.setAttribute('aria-pressed', 'false');
      el.graphContainer.classList.add('hidden');
      el.conceptListView.classList.remove('hidden');
    });

    // Zoom Controls
    el.btnZoomIn.addEventListener('click', () => {
      state.zoom = Math.min(2.5, state.zoom * 1.2);
      updateViewportTransform();
    });

    el.btnZoomOut.addEventListener('click', () => {
      state.zoom = Math.max(0.4, state.zoom / 1.2);
      updateViewportTransform();
    });

    el.btnZoomReset.addEventListener('click', () => {
      state.zoom = 1.0;
      state.pan = { x: 60, y: 40 };
      updateViewportTransform();
    });

    // Pan Controls (Mouse Drag on Graph)
    el.graphContainer.addEventListener('mousedown', (e) => {
      if (e.target.closest('.node-group')) return; // Allow node clicks
      state.isDragging = true;
      state.dragStart = { x: e.clientX - state.pan.x, y: e.clientY - state.pan.y };
    });

    window.addEventListener('mousemove', (e) => {
      if (!state.isDragging) return;
      state.pan.x = e.clientX - state.dragStart.x;
      state.pan.y = e.clientY - state.dragStart.y;
      updateViewportTransform();
    });

    window.addEventListener('mouseup', () => {
      state.isDragging = false;
    });

    // Wheel Zoom
    el.graphContainer.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      state.zoom = Math.max(0.4, Math.min(2.5, state.zoom * zoomFactor));
      updateViewportTransform();
    });

    // Evidence Transparency Accordion Toggle
    el.evidenceAccordionToggle.addEventListener('click', () => {
      const isExpanded = el.evidenceAccordionToggle.getAttribute('aria-expanded') === 'true';
      el.evidenceAccordionToggle.setAttribute('aria-expanded', !isExpanded);
      el.evidenceAccordionBody.classList.toggle('open');
    });

    // Assessment Modal Listeners
    el.btnCloseAssessment.addEventListener('click', closeAssessmentModal);
    el.btnFinishAssessment.addEventListener('click', closeAssessmentModal);
    el.btnSubmitAssessmentAnswer.addEventListener('click', submitAssessmentAnswer);
    el.assessmentLangSelect.addEventListener('change', (e) => {
      state.currentLanguage = e.target.value;
      if (state.activeSession && state.currentAssessmentItem) {
        launchAssessment(state.currentAssessmentItem.concept_id);
      }
    });

    // Source Modal Listeners
    el.btnCloseSource.addEventListener('click', () => {
      el.sourceModal.classList.add('hidden');
    });

    // Upload Modal Listeners
    el.btnOpenUploadModal.addEventListener('click', () => {
      el.uploadModal.classList.remove('hidden');
      el.uploadFileInput.value = '';
      if (el.uploadSubjectInput) el.uploadSubjectInput.value = '';
      el.uploadStatusBox.classList.add('hidden');
      el.btnSubmitUpload.disabled = false;
      el.btnSubmitUpload.textContent = '🚀 Ingest & Build Knowledge Map';
    });

    el.btnCloseUpload.addEventListener('click', () => {
      el.uploadModal.classList.add('hidden');
    });

    // Auto-suggest topic/paper name when a file is selected
    el.uploadFileInput.addEventListener('change', () => {
      const file = el.uploadFileInput.files[0];
      if (file && el.uploadSubjectInput && (!el.uploadSubjectInput.value || el.uploadSubjectInput.value.trim() === '')) {
        const base = file.name.replace(/\.[^/.]+$/, '');
        const cleaned = base.replace(/[-_]+/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        el.uploadSubjectInput.value = cleaned;
      }
    });

    el.btnSubmitUpload.addEventListener('click', async () => {
      const file = el.uploadFileInput.files[0];
      if (!file) {
        showToast('Please select a document file (.pptx, .pdf, .txt, .md).');
        return;
      }

      const rawSubject = (el.uploadSubjectInput ? el.uploadSubjectInput.value : '').trim();
      const subjectName = rawSubject || file.name.replace(/\.[^/.]+$/, '').replace(/[-_]+/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

      const formData = new FormData();
      formData.append('file', file);
      formData.append('subject_name', subjectName);
      formData.append('subject_id', subjectName);

      try {
        el.btnSubmitUpload.disabled = true;
        el.btnSubmitUpload.textContent = '⏳ Extracting & Indexing...';
        el.uploadStatusBox.classList.remove('hidden');
        el.uploadStatusText.textContent = `Processing "${file.name}" for "${subjectName}"... Extracting content, generating chunks, and mapping concepts.`;

        const res = await fetch('/api/documents/upload', {
          method: 'POST',
          body: formData
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Upload processing failed.');
        }

        showToast(`✓ Ingested into "${data.subject_name}": ${data.pages_extracted} slides/pages, ${data.chunks_indexed} chunks, ${data.concepts_discovered} concepts mapped!`);
        el.uploadModal.classList.add('hidden');

        // Automatically switch active topic to the newly created one!
        state.subjectId = data.subject_id;
        state.selectedConceptId = null;

        // Re-populate the subjects dropdown and select the new topic
        await populateSubjects();
        el.subjectSelect.value = state.subjectId;

        // Reload dashboard for the new topic
        await loadDashboard();
      } catch (err) {
        console.error(err);
        el.uploadStatusBox.classList.remove('hidden');
        el.uploadStatusText.textContent = `Error: ${err.message}`;
        showToast(`Upload failed: ${err.message}`);
      } finally {
        el.btnSubmitUpload.disabled = false;
        el.btnSubmitUpload.textContent = '🚀 Ingest & Build Knowledge Map';
      }
    });

    // Close Modals on Backdrop Click
    window.addEventListener('click', (e) => {
      if (e.target === el.assessmentModal) closeAssessmentModal();
      if (e.target === el.sourceModal) el.sourceModal.classList.add('hidden');
      if (e.target === el.uploadModal) el.uploadModal.classList.add('hidden');
    });
  }

  // =========================================================================
  // Initialization
  // =========================================================================

  async function init() {
    setupEventListeners();
    await populateSubjects();
    await loadDashboard();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
