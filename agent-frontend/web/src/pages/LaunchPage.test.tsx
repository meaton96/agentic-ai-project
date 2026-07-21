import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { LaunchPage } from './LaunchPage'
import * as client from '../api/client'
import type { DatasetInfo } from '../api/types'

// Partial mock: keeps the real ApiError class (LaunchPage's catch block does
// `e instanceof ApiError`, which would never match an auto-mocked class),
// only replacing the two functions this page actually calls.
vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return { ...actual, listDatasets: vi.fn(), launchRun: vi.fn() }
})

const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return { ...actual, useNavigate: () => mockNavigate }
})

const DATASETS: DatasetInfo[] = [{ filename: 'churn.csv', size: 100, modified: 0 }]

describe('LaunchPage', () => {
  beforeEach(() => {
    mockNavigate.mockReset()
    vi.mocked(client.listDatasets).mockResolvedValue(DATASETS)
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
    })
    expect(mockNavigate).toHaveBeenCalledWith('/runs/run_new123')
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
