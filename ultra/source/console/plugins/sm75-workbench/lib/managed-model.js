// The managed local provider follows the active engine. Historical request
// headers stay unchanged; only the next request resolves to the current model.
let active;
export function setManagedModel(selection) {
  if (selection?.provider !== 'sm75-local' || !selection.model) throw Error('Invalid managed model');
  active = {provider:selection.provider,model:selection.model};
}
export function resolveManagedSelection(selection, current = active) {
  if (!current || selection?.provider !== 'sm75-local' || selection.model === current.model) return selection;
  // Reasoning controls belong to the old model and cannot cross model boundaries.
  return {provider:current.provider,model:current.model};
}
