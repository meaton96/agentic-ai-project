import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { PromptsPage } from './PromptsPage'
import * as client from '../api/client'
import type { PromptInfo } from '../api/types'

vi.mock('../api/client')

const PROMPTS: PromptInfo[] = [
  { agent: 'modeling', default_content: 'default modeling text', override_content: null, has_override: false },
  { agent: 'profiler', default_content: 'default profiler text', override_content: 'custom profiler text', has_override: true },
]

describe('PromptsPage', () => {
  it('renders one card per agent, showing default content and override-active only where applicable', async () => {
    vi.mocked(client.listPrompts).mockResolvedValue(PROMPTS)
    render(<PromptsPage />)

    const modelingHeading = await screen.findByRole('heading', { name: 'modeling' })
    const modelingCard = modelingHeading.closest('.card') as HTMLElement
    expect(within(modelingCard).getByTestId('modeling-default')).toHaveTextContent('default modeling text')
    expect(within(modelingCard).queryByText('override active')).not.toBeInTheDocument()
    expect(within(modelingCard).getByRole('button', { name: 'Revert to default' })).toBeDisabled()

    const profilerCard = screen.getByRole('heading', { name: 'profiler' }).closest('.card') as HTMLElement
    expect(within(profilerCard).getByText('override active')).toBeInTheDocument()
    expect(within(profilerCard).getByRole('button', { name: 'Revert to default' })).toBeEnabled()
    expect(within(profilerCard).getByLabelText('profiler override')).toHaveValue('custom profiler text')
  })

  it('saving an edited override calls the API and flips the card to override-active', async () => {
    vi.mocked(client.listPrompts).mockResolvedValue(PROMPTS)
    vi.mocked(client.putPromptOverride).mockResolvedValue({
      agent: 'modeling', default_content: 'default modeling text', override_content: 'my new modeling prompt', has_override: true,
    })
    const user = userEvent.setup()
    render(<PromptsPage />)

    const modelingCard = (await screen.findByRole('heading', { name: 'modeling' })).closest('.card') as HTMLElement
    const textarea = within(modelingCard).getByLabelText('modeling override')
    await user.clear(textarea)
    await user.type(textarea, 'my new modeling prompt')

    const saveButton = within(modelingCard).getByRole('button', { name: 'Save override' })
    expect(saveButton).toBeEnabled()
    await user.click(saveButton)

    await waitFor(() => expect(client.putPromptOverride).toHaveBeenCalledWith('modeling', 'my new modeling prompt'))
    expect(within(modelingCard).getByText('override active')).toBeInTheDocument()
  })

  it('reverting deletes the override and resets the textarea to the shipped default', async () => {
    vi.mocked(client.listPrompts).mockResolvedValue(PROMPTS)
    vi.mocked(client.deletePromptOverride).mockResolvedValue({
      agent: 'profiler', default_content: 'default profiler text', override_content: null, has_override: false,
    })
    const user = userEvent.setup()
    render(<PromptsPage />)

    const profilerCard = (await screen.findByRole('heading', { name: 'profiler' })).closest('.card') as HTMLElement
    await user.click(within(profilerCard).getByRole('button', { name: 'Revert to default' }))

    await waitFor(() => expect(client.deletePromptOverride).toHaveBeenCalledWith('profiler'))
    expect(within(profilerCard).queryByText('override active')).not.toBeInTheDocument()
    expect(within(profilerCard).getByLabelText('profiler override')).toHaveValue('default profiler text')
  })
})
