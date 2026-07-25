import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { LaunchPage } from './LaunchPage'
import * as client from '../api/client'
import type { DatasetInfo, PromptInfo } from '../api/types'

// Partial mock: keeps the real ApiError class (LaunchPage's catch block does
// `e instanceof ApiError`, which would never match an auto-mocked class),
// only replacing the functions this page actually calls.
vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return { ...actual, listDatasets: vi.fn(), listPrompts: vi.fn(), launchRun: vi.fn() }
})

const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return { ...actual, useNavigate: () => mockNavigate }
})

const DATASETS: DatasetInfo[] = [{ filename: 'churn.csv', size: 100, modified: 0 }]
const NO_OVERRIDES: PromptInfo[] = [
  { agent: 'profiler', default_content: 'default text', override_content: null, has_override: false },
]

describe('LaunchPage', () => {
  beforeEach(() => {
    mockNavigate.mockReset()
    vi.mocked(client.launchRun).mockReset()
    vi.mocked(client.listDatasets).mockResolvedValue(DATASETS)
    vi.mocked(client.listPrompts).mockResolvedValue(NO_OVERRIDES)
  })

  it('disables goal once target is filled in, and vice versa', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <LaunchPage />
      </MemoryRouter>,
    )
    await screen.findByDisplayValue('churn.csv')

    const target = screen.getByLabelText(/Target column/)
    const goal = screen.getByLabelText(/Goal/)
    expect(target).toBeEnabled()
    expect(goal).toBeEnabled()

    await user.type(target, 'churned')
    expect(goal).toBeDisabled()

    await user.clear(target)
    expect(goal).toBeEnabled()

    await user.type(goal, 'predict churn')
    expect(target).toBeDisabled()
  })

  it('posts the expected payload shape and navigates to the new run', async () => {
    vi.mocked(client.launchRun).mockResolvedValue({ run_id: 'run_new123', status: 'running' })
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <LaunchPage />
      </MemoryRouter>,
    )
    await screen.findByDisplayValue('churn.csv')

    await user.type(screen.getByLabelText(/Target column/), 'churned')
    await user.click(screen.getByLabelText('dynamic'))
    await user.click(screen.getByRole('button', { name: /Launch run/ }))

    await waitFor(() => expect(client.launchRun).toHaveBeenCalledTimes(1))
    expect(client.launchRun).toHaveBeenCalledWith({
      dataset: 'churn.csv',
      orchestrator: 'dynamic',
      target: 'churned',
      goal: undefined,
      strategy: undefined,
      max_candidates: undefined,
      model_endpoint: 'rit',
      skip_feature_engineering: false,
      use_prompt_overrides: false,
      use_mcp: false,
    })
    expect(mockNavigate).toHaveBeenCalledWith('/runs/run_new123')
  })

  it('disables the prompt-overrides toggle when no agent has one configured, enables it when one does', async () => {
    render(
      <MemoryRouter>
        <LaunchPage />
      </MemoryRouter>,
    )
    await screen.findByDisplayValue('churn.csv')
    expect(screen.getByLabelText(/Use prompt overrides/)).toBeDisabled()
  })

  it('enables the prompt-overrides toggle and includes it in the payload when an override exists', async () => {
    vi.mocked(client.listPrompts).mockResolvedValue([
      { agent: 'modeling', default_content: 'default', override_content: 'custom', has_override: true },
    ])
    vi.mocked(client.launchRun).mockResolvedValue({ run_id: 'run_new456', status: 'running' })
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <LaunchPage />
      </MemoryRouter>,
    )
    await screen.findByDisplayValue('churn.csv')
    const toggle = await screen.findByLabelText(/Use prompt overrides/)
    await waitFor(() => expect(toggle).toBeEnabled())

    await user.type(screen.getByLabelText(/Target column/), 'churned')
    await user.click(toggle)
    await user.click(screen.getByRole('button', { name: /Launch run/ }))

    await waitFor(() => expect(client.launchRun).toHaveBeenCalledTimes(1))
    expect(client.launchRun).toHaveBeenCalledWith(expect.objectContaining({ use_prompt_overrides: true }))
  })

  it('disables the MCP toggle for the static orchestrator and enables it for dynamic', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <LaunchPage />
      </MemoryRouter>,
    )
    await screen.findByDisplayValue('churn.csv')

    const mcpToggle = screen.getByLabelText(/Fetch tool facts via MCP/)
    expect(mcpToggle).toBeDisabled()

    await user.click(screen.getByLabelText('dynamic'))
    expect(mcpToggle).toBeEnabled()
  })

  it('includes use_mcp: true in the payload when checked on a dynamic run', async () => {
    vi.mocked(client.launchRun).mockResolvedValue({ run_id: 'run_new789', status: 'running' })
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <LaunchPage />
      </MemoryRouter>,
    )
    await screen.findByDisplayValue('churn.csv')

    await user.click(screen.getByLabelText('dynamic'))
    await user.click(screen.getByLabelText(/Fetch tool facts via MCP/))
    await user.type(screen.getByLabelText(/Goal/), 'predict churn')
    await user.click(screen.getByRole('button', { name: /Launch run/ }))

    await waitFor(() => expect(client.launchRun).toHaveBeenCalledTimes(1))
    expect(client.launchRun).toHaveBeenCalledWith(expect.objectContaining({ use_mcp: true }))
  })

  it('shows an error and does not navigate if launching fails', async () => {
    vi.mocked(client.launchRun).mockRejectedValue(new client.ApiError(422, "dataset 'x' does not exist"))
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <LaunchPage />
      </MemoryRouter>,
    )
    await screen.findByDisplayValue('churn.csv')
    await user.type(screen.getByLabelText(/Target column/), 'churned')
    await user.click(screen.getByRole('button', { name: /Launch run/ }))

    expect(await screen.findByText(/422:.*does not exist/)).toBeInTheDocument()
    expect(mockNavigate).not.toHaveBeenCalled()
  })
})
