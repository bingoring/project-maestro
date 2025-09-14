import React, { useState, useCallback, useMemo } from 'react'
import {
  TextArea,
  Button,
  Tag,
  InlineNotification,
  FileUploader,
  ProgressIndicator,
  ProgressStep,
  Stack,
  Tile,
  FormGroup,
} from '@carbon/react'
import { Send, Add, TrashCan, Information } from '@carbon/icons-react'
import { useWorkflowAPI } from '@hooks/useWorkflowAPI'
import { useWebSocket } from '@hooks/useWebSocket'
import type { PromptData, PromptFormData } from '@types'
import './PromptInterface.scss'

interface PromptInterfaceProps {
  onSubmit?: (prompt: PromptData) => void
  onWorkflowStart?: (workflowId: string) => void
  className?: string
}

export const PromptInterface: React.FC<PromptInterfaceProps> = ({
  onSubmit,
  onWorkflowStart,
  className = '',
}) => {
  const [formData, setFormData] = useState<PromptFormData>({
    prompt: '',
    files: [],
    complexity: 'moderate',
  })
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  
  const { submitPrompt, loading: apiLoading, error: apiError } = useWorkflowAPI()
  const { sendMessage, isConnected } = useWebSocket(
    'ws://localhost:8000/ws',
    'user-1' // TODO: Get from auth context
  )

  // Analyze prompt complexity in real-time
  const complexity = useMemo(() => {
    const { prompt } = formData
    if (!prompt.trim()) return 'simple'
    
    const wordCount = prompt.split(/\\s+/).length
    const hasCode = /```[\\s\\S]*?```|`[^`]+`/.test(prompt)
    const hasMultipleSteps = /(\\d+\\.|step|then|after|first|second|third|next|finally)/gi.test(prompt)
    const hasQuestions = /\\?/g.test(prompt)
    const questionCount = (prompt.match(/\\?/g) || []).length
    
    let complexityScore = 0
    
    // Word count scoring
    if (wordCount > 100) complexityScore += 3
    else if (wordCount > 50) complexityScore += 2
    else if (wordCount > 20) complexityScore += 1
    
    // Code presence
    if (hasCode) complexityScore += 2
    
    // Multiple steps
    if (hasMultipleSteps) complexityScore += 2
    
    // Questions
    if (questionCount > 2) complexityScore += 2
    else if (questionCount > 0) complexityScore += 1
    
    // File context
    if (formData.files.length > 0) complexityScore += 1
    
    if (complexityScore >= 5) return 'complex'
    if (complexityScore >= 2) return 'moderate'
    return 'simple'
  }, [formData.prompt, formData.files.length])

  const handlePromptChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setFormData(prev => ({ ...prev, prompt: e.target.value }))
  }, [])

  const handleFilesChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    setFormData(prev => ({ ...prev, files }))
  }, [])

  const removeFile = useCallback((index: number) => {
    setFormData(prev => ({
      ...prev,
      files: prev.files.filter((_, i) => i !== index)
    }))
  }, [])

  const clearForm = useCallback(() => {
    setFormData({
      prompt: '',
      files: [],
      complexity: 'moderate',
    })
    setCurrentStep(0)
  }, [])

  const handleSubmit = useCallback(async () => {
    if (!formData.prompt.trim()) return

    setIsProcessing(true)
    setCurrentStep(1)

    try {
      const promptData: PromptData = {
        prompt: formData.prompt,
        context: formData.files,
        complexity,
        timestamp: new Date().toISOString(),
        metadata: {
          wordCount: formData.prompt.split(/\\s+/).length,
          fileCount: formData.files.length,
        },
      }

      // Notify parent component
      onSubmit?.(promptData)

      // Send to WebSocket for real-time updates
      if (isConnected) {
        sendMessage({
          type: 'prompt_submission',
          data: promptData,
        })
      }

      // Submit via API
      setCurrentStep(2)
      const response = await submitPrompt(formData.prompt, {
        files: formData.files.map(f => ({ name: f.name, size: f.size, type: f.type })),
        complexity,
      })

      setCurrentStep(3)

      if (response.success && response.data?.workflowId) {
        onWorkflowStart?.(response.data.workflowId)
        // Clear form after successful submission
        setTimeout(() => {
          clearForm()
          setIsProcessing(false)
        }, 1000)
      } else {
        throw new Error(response.error || 'Failed to submit prompt')
      }

    } catch (error) {
      console.error('Failed to submit prompt:', error)
      setIsProcessing(false)
      setCurrentStep(0)
    }
  }, [formData, complexity, onSubmit, onWorkflowStart, isConnected, sendMessage, submitPrompt, clearForm])

  const getComplexityTagType = (complexity: PromptData['complexity']) => {
    switch (complexity) {
      case 'simple': return 'green'
      case 'moderate': return 'blue'
      case 'complex': return 'red'
      default: return 'gray'
    }
  }

  const isSubmitDisabled = !formData.prompt.trim() || isProcessing || apiLoading

  return (
    <div className={`prompt-interface ${className}`}>
      <Tile className="prompt-tile">
        <div className="prompt-header">
          <h2>AI Assistant Prompt</h2>
          <div className="complexity-indicator">
            <Tag type={getComplexityTagType(complexity)} size="sm">
              {complexity} query
            </Tag>
            {!isConnected && (
              <Tag type="red" size="sm">
                Offline
              </Tag>
            )}
          </div>
        </div>

        {isProcessing && (
          <div className="progress-section">
            <ProgressIndicator currentIndex={currentStep} spaceEqually>
              <ProgressStep
                label="Preparing"
                description="Analyzing prompt"
                complete={currentStep > 0}
                current={currentStep === 0}
              />
              <ProgressStep
                label="Submitting"
                description="Sending to AI agents"
                complete={currentStep > 1}
                current={currentStep === 1}
              />
              <ProgressStep
                label="Processing"
                description="Creating workflow"
                complete={currentStep > 2}
                current={currentStep === 2}
              />
              <ProgressStep
                label="Complete"
                description="Workflow started"
                complete={currentStep >= 3}
                current={currentStep === 3}
              />
            </ProgressIndicator>
          </div>
        )}

        <FormGroup>
          <Stack gap={4}>
            <TextArea
              id="prompt-input"
              labelText="Describe what you want to build or achieve"
              placeholder="Example: Create a 2D platformer game with physics-based movement, collectible items, and a scoring system. The player should be able to jump, run, and interact with objects in the environment."
              rows={6}
              value={formData.prompt}
              onChange={handlePromptChange}
              disabled={isProcessing}
              invalid={!!apiError}
              invalidText={apiError || undefined}
            />

            <div className="file-upload-section">
              <FileUploader
                labelTitle="Add Context Files"
                labelDescription="Upload relevant files to provide additional context (max 10MB each)"
                buttonLabel="Browse files"
                filenameStatus="edit"
                accept={['.txt', '.md', '.json', '.yaml', '.yml', '.py', '.js', '.ts', '.jsx', '.tsx']}
                multiple
                onChange={handleFilesChange}
                disabled={isProcessing}
              />
              
              {formData.files.length > 0 && (
                <div className="uploaded-files">
                  <h4>Uploaded files:</h4>
                  <Stack gap={2}>
                    {formData.files.map((file, index) => (
                      <div key={index} className="file-item">
                        <span>{file.name}</span>
                        <span className="file-size">({(file.size / 1024).toFixed(1)} KB)</span>
                        <Button
                          kind="ghost"
                          size="sm"
                          hasIconOnly
                          iconDescription="Remove file"
                          renderIcon={TrashCan}
                          onClick={() => removeFile(index)}
                          disabled={isProcessing}
                        />
                      </div>
                    ))}
                  </Stack>
                </div>
              )}
            </div>

            <div className="prompt-actions">
              <Button
                kind="primary"
                onClick={handleSubmit}
                disabled={isSubmitDisabled}
                renderIcon={Send}
              >
                {isProcessing ? 'Processing...' : 'Submit Request'}
              </Button>

              <Button
                kind="secondary"
                onClick={clearForm}
                disabled={isProcessing}
                renderIcon={TrashCan}
              >
                Clear
              </Button>
            </div>

            {complexity === 'complex' && (
              <InlineNotification
                kind="info"
                title="Complex Query Detected"
                subtitle="This request may take longer to process and will use multiple AI agents."
                hideCloseButton
                lowContrast
              />
            )}

            {apiError && (
              <InlineNotification
                kind="error"
                title="Submission Error"
                subtitle={apiError}
                hideCloseButton
              />
            )}

            {isProcessing && (
              <InlineNotification
                kind="info"
                title="Processing Request"
                subtitle="Your request is being processed by the AI agents. You can monitor progress in the workflow dashboard."
                hideCloseButton
              />
            )}
          </Stack>
        </FormGroup>
      </Tile>
    </div>
  )
}